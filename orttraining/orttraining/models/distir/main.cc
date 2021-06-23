// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/util/math.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/profiler.h"
#include "core/session/environment.h"
#include "core/framework/bfc_arena.h"
#include "core/framework/random_seed.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#ifdef USE_CUDA
namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);
}  // namespace onnxruntime
#endif
#ifdef USE_ROCM
#include "core/providers/rocm/rocm_allocator.h"
#include "core/providers/rocm/rocm_provider_factory_creator.h"
#endif
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/models/runner/constant.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"
#include "orttraining/models/runner/data_loader.h"
#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
#include "orttraining/training_ops/cuda/communication/nccl_service.h"
#endif

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

static SessionOptions session_options = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
    ExecutionOrder::PRIORITY_BASED,    //execution_order
    false,                             //enable_profiling
    ORT_TSTR(""),                      //optimized_model_filepath
    true,                              //enable_mem_pattern
    true,                              //enable_mem_reuse
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level1,          //graph_optimization_level
    {},                                //intra_op_param
    {},                                //inter_op_param
    {},                                //free_dimension_overrides
    true,                              //use_per_session_threads
    true,                              //thread_pool_allow_spinning
    false,                             //use_deterministic_compute
    {},                                //config_options
    {},                                // initializers_to_share_map
};

// int main(int argc, char* argv[]) {
int main() {
  // setup logger, be noted: LOGS_DEFAULT must be after logging manager initialization.
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{
      unique_ptr<logging::ISink>{new logging::CLogSink{}},
      logging::Severity::kWARNING,
      false,
      logging::LoggingManager::InstanceType::Default,
      &default_logger_id,
      -1};

  auto world_size = MPIContext::GetInstance().GetWorldSize();
  assert(MPIContext::GetInstance().GetWorldRank() == MPIContext::GetInstance().GetLocalRank());
  auto rank = MPIContext::GetInstance().GetWorldRank();
  cout << "World size: " << world_size << endl;

  // For debugging mpiruns:
  // {
  //   volatile int i = 0;
  //   printf("Rank %d PID %d ready for attach\n", rank, getpid());
  //   fflush(stdout);
  //   while (0 == i)
  //     sleep(5);
  // }

  // Set up environment
  unique_ptr<Environment> env;
  RETURN_IF_FAIL(Environment::Create(nullptr, env));

  // Create inference session
  InferenceSession inference_session{session_options, *env};

  // Load the .onnx file
  if (world_size > 1) {
    std::ostringstream filename;
    filename << "/home/t-sikris/onnxruntime/model-" << rank << ".onnx";
    ORT_THROW_IF_ERROR(inference_session.Load(filename.str()));
  } else {
    ORT_THROW_IF_ERROR(inference_session.Load("/home/t-sikris/onnxruntime/model.onnx"));
  }

  // Register execution provider
#ifdef USE_CUDA
  {
    OrtCUDAProviderOptions info{
        gsl::narrow<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank()),
        OrtCudnnConvAlgoSearch::EXHAUSTIVE,
        std::numeric_limits<size_t>::max(),
        0,
        true,
        0,
        nullptr,
        nullptr};

    auto factory = CreateExecutionProviderFactory_Cuda(&info);
    auto input_allocator = CreateCUDAPinnedAllocator(info.device_id, CUDA_PINNED);
    std::unique_ptr<IExecutionProvider> provider = factory->CreateProvider();
    ORT_THROW_IF_ERROR(inference_session.RegisterExecutionProvider(std::move(provider)));
  }
#endif

  // Initialize process groups for collective communications
  DistributedRunContext::CreateInstance({rank,
                                         world_size,
                                         rank,
                                         world_size,
                                         2,
                                         1,
                                         1});

  ORT_THROW_IF_ERROR(inference_session.Initialize());

  // Create random input data -- TODO get from .onnx file/external file?
  VectorString feed_names;
  VectorString fetch_names;
  std::vector<MLValue> feeds;
  std::vector<MLValue> fetches = std::vector<MLValue>();
  if (false) {
    // Add/ReluGrad models
    MLValue xValue;
    std::vector<float> xVals(2, 2.0);
    TrainingUtil::CreateCpuMLValue({1, 2}, xVals, &xValue);
    feed_names = {"X"};
    fetch_names = {"Z"};
    feeds = {xValue};
  } else if (false) {
    // Send/recv models
    MLValue signalValue;
    TrainingUtil::CreateCpuMLScalar(true, &signalValue);
    MLValue srcValue;
    int64_t srcRank = 0;
    TrainingUtil::CreateCpuMLScalar(srcRank, &srcValue);
    MLValue dstValue;
    int64_t dstRank = 1;
    TrainingUtil::CreateCpuMLScalar(dstRank, &dstValue);
    MLValue xValue;
    TrainingUtil::CreateCpuMLValue({1, 2}, std::vector<float>{1.0, 4.0}, &xValue);
    fetch_names = {"output_signal"};
    if (rank == 0) {
      feed_names = {"input_signal_token", "dst_rank_token", "X"};
      feeds = {signalValue, dstValue, xValue};
      fetches = std::vector<MLValue>();
    } else {
      feed_names = {"input_signal_token", "src_rank_token"};
      feeds = {signalValue, srcValue};
      fetches = std::vector<MLValue>();
    }
  } else {
    // Allreduce models
    MLValue xValue;
    std::vector<float> xVals(2, 2.0);
    TrainingUtil::CreateCpuMLValue({1, 2}, xVals, &xValue);
    feed_names = {"X"};
    fetch_names = {"Z"};
    feeds = {xValue};
  }

  // Create communication plan.
#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  // // TODO scan graph and get plan? Or get from file?
  // auto& nccl_service = cuda::INcclService::GetInstance();

  // nccl_service.PlanStart();
  // nccl_service.PlanNewGroupStart();
  // if (rank == 0) {
  //   nccl_service.PlanSend(1);
  // } else {
  //   nccl_service.PlanRecv(0);
  // }
  // nccl_service.PlanNewGroupEnd();
  // nccl_service.PlanEnd();

  // // Launch NCCL service to execute the plan.
  // nccl_service.Launch();
#endif

  // Run the file:
  RunOptions run_options;
  run_options.only_execute_path_to_fetches = true;
  // common::Status status = inference_session.Run(
  ORT_THROW_IF_ERROR(inference_session.Run(
      run_options,
      feed_names,
      feeds,
      fetch_names,
      &fetches));
  // status = inference_session.PartialRun(run_options, feeds, fetches, state,
  //   feeds_fetches_manager);

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  // nccl_service.Reset();
  // nccl_service.Terminate();
#endif

  // To see inputs/outputs, build with:
  // --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
  // and use env variables:
  // ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1 ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1

#if defined(USE_MPI)
#ifdef _WIN32
  // https://docs.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-best-practices
  // shutdown_mpi() is not called within MPIContext destructor because of DllMain's restriction
  // call shutdown_mpi() here instead.
  MPIContext::shutdown_mpi();
#endif
#endif
  return 0;
}
