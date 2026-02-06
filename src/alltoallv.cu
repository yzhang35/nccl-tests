/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"
#include <vector>

#pragma weak ncclAlltoAllv
#pragma weak ncclCommQueryProperties

namespace {
struct AlltoAllvPlan {
  size_t count = 0;
  int nranks = 0;
  std::vector<size_t> sendcounts;
  std::vector<size_t> sdispls;
  std::vector<size_t> recvcounts;
  std::vector<size_t> rdispls;
};

static thread_local AlltoAllvPlan plan;
static thread_local void** relaybuffsForThread = nullptr;
static thread_local int relaybuffsFirstRank = -1;
static thread_local int relaybuffsCount = 0;

// Generate balanced counts and displacements for alltoallv
static void BuildBalancedCounts(size_t totalCount, int nranks, AlltoAllvPlan* out) {
  out->count = totalCount;
  out->nranks = nranks;
  const size_t n = (size_t)nranks;
  out->sendcounts.assign(n * n, 0);
  out->sdispls.assign(n * n, 0);
  out->recvcounts.assign(n * n, 0);
  out->rdispls.assign(n * n, 0);

  const size_t base = n ? (totalCount / n) : 0;
  const size_t rem = n ? (totalCount % n) : 0;

  for (int r = 0; r < nranks; ++r) {
    const size_t row = (size_t)r * n;
    for (int p = 0; p < nranks; ++p) {
      out->sendcounts[row + (size_t)p] = base;
    }
    for (size_t p = 0; p < rem; ++p) {
      int peer = (int)((p + (size_t)r) % n);
      out->sendcounts[row + (size_t)peer] += 1;
    }
    size_t off = 0;
    for (int p = 0; p < nranks; ++p) {
      out->sdispls[row + (size_t)p] = off;
      off += out->sendcounts[row + (size_t)p];
    }
  }

  for (int r = 0; r < nranks; ++r) {
    const size_t row = (size_t)r * n;
    for (int p = 0; p < nranks; ++p) {
      out->recvcounts[row + (size_t)p] = out->sendcounts[(size_t)p * n + (size_t)r];
    }
    size_t off = 0;
    for (int p = 0; p < nranks; ++p) {
      out->rdispls[row + (size_t)p] = off;
      off += out->recvcounts[row + (size_t)p];
    }
  }
}

// Generate unbalanced counts and displacements for alltoallv
static void BuildUnbalancedCounts(size_t totalCount, int nranks, AlltoAllvPlan* out) {
  out->count = totalCount;
  out->nranks = nranks;
  const size_t n = (size_t)nranks;
  out->sendcounts.assign(n * n, 0);
  out->sdispls.assign(n * n, 0);
  out->recvcounts.assign(n * n, 0);
  out->rdispls.assign(n * n, 0);

  if (n == 0) return;

  const size_t sumWeights = n * (n + 1) / 2;
  const size_t div = totalCount / sumWeights;
  const size_t rem = totalCount - div * sumWeights;

  size_t used = 0;
  for (size_t p = 0; p < n; ++p) {
    const size_t w = p + 1;
    const size_t v = div * w + (rem * w) / sumWeights;
    out->sendcounts[p] = v;
    used += v;
  }
  size_t leftover = totalCount - used;
  for (size_t i = 0; i < leftover; ++i) {
    out->sendcounts[n - 1 - i] += 1;
  }

  size_t off0 = 0;
  for (size_t p = 0; p < n; ++p) {
    out->sdispls[p] = off0;
    off0 += out->sendcounts[p];
  }

  for (int r = 1; r < nranks; ++r) {
    const size_t row = (size_t)r * n;
    const size_t shift = (size_t)r;
    const size_t head = n - shift;
    for (size_t p = 0; p < shift; ++p) {
      out->sendcounts[row + p] = out->sendcounts[head + p];
    }
    for (size_t p = 0; p < head; ++p) {
      out->sendcounts[row + shift + p] = out->sendcounts[p];
    }
    size_t off = 0;
    for (size_t p = 0; p < n; ++p) {
      out->sdispls[row + p] = off;
      off += out->sendcounts[row + p];
    }
  }

  for (int r = 0; r < nranks; ++r) {
    const size_t row = (size_t)r * n;
    for (int p = 0; p < nranks; ++p) {
      out->recvcounts[row + (size_t)p] = out->sendcounts[(size_t)p * n + (size_t)r];
    }
    size_t off = 0;
    for (int p = 0; p < nranks; ++p) {
      out->rdispls[row + (size_t)p] = off;
      off += out->recvcounts[row + (size_t)p];
    }
  }
}

static AlltoAllvPlan* GetPlan(size_t totalCount, int nranks) {
  if (plan.count != totalCount || plan.nranks != nranks) {
    // Use Unbalanced counts for correctness testing
    //BuildUnbalancedCounts(totalCount, nranks, &plan);
    // Use Balanced counts for bandwidth testing
    BuildBalancedCounts(totalCount, nranks, &plan);
  }
  return &plan;
}
}

void AlltoAllvGetCollByteCount(size_t *sendcount, size_t *recvcount, size_t *paramcount, size_t *sendInplaceOffset, size_t *recvInplaceOffset, size_t count, size_t eltSize, int nranks) {
  (void)eltSize;
  (void)nranks;
  *sendcount = count;
  *recvcount = count;
  *sendInplaceOffset = 0;
  *recvInplaceOffset = 0;
  *paramcount = count;
}

testResult_t AlltoAllvInitData(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int rep, int in_place) {
  (void)op;
  (void)root;
  size_t sendcount = args->sendBytes / wordSize(type);
  size_t recvcount = args->expectedBytes / wordSize(type);
  int nranks = args->nProcs*args->nThreads*args->nGpus;

  if (sendcount != recvcount) return testInternalError;
  AlltoAllvPlan* planPtr = GetPlan(sendcount, nranks);

  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));
    int rank = ((args->proc*args->nThreads + args->thread)*args->nGpus + i);
    CUDACHECK(cudaMemset(args->recvbuffs[i], 0, args->expectedBytes));

    void* data = in_place ? args->recvbuffs[i] : args->sendbuffs[i];
    // Pseudo-random but deterministic per rank and repetition
    TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33*rep + rank, 1, 0));

    // Build expected receive buffer from each peer's send data
    const size_t n = (size_t)nranks;
    const size_t row = (size_t)rank * n;
    for (int src = 0; src < nranks; src++) {
      size_t cnt = planPtr->recvcounts[row + (size_t)src];
      if (cnt == 0) continue;
      size_t rdisp = planPtr->rdispls[row + (size_t)src];
      size_t sdisp = planPtr->sdispls[(size_t)src * n + (size_t)rank];
      TESTCHECK(InitData((char*)args->expected[i] + rdisp * wordSize(type),
                         cnt, sdisp, type, ncclSum, 33*rep + src, 1, 0));
    }
    CUDACHECK(cudaDeviceSynchronize());
  }

  // We don't support in-place alltoallv
  args->reportErrors = in_place ? 0 : 1;
  return testSuccess;
}

void AlltoAllvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t AlltoAllvRunColl(void* sendbuff, size_t sendoffset, void* recvbuff, size_t recvoffset, size_t count, ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream, int deviceImpl) {
  (void)op;
  (void)root;
  if (deviceImpl != 0) return testNotImplemented;

  if (ncclAlltoAllv == nullptr) {
    if (is_main_thread) {
      fprintf(stderr, "AlltoAllv: ncclAlltoAllv symbol not found. Make sure you are using a VCCL_MOE build.\n");
    }
    return testNotImplemented;
  }

  int nranks = 0;
  NCCLCHECK(ncclCommCount(comm, &nranks));

  AlltoAllvPlan* planPtr = GetPlan(count, nranks);

  char* sptr = (char*)sendbuff + sendoffset;
  char* rptr = (char*)recvbuff + recvoffset;

  void* relaybuff = nullptr;
  size_t relayCount = 0;
  if (use_relay_buffer) {
    int rank = 0;
    NCCLCHECK(ncclCommUserRank(comm, &rank));
    int localIndex = rank - relaybuffsFirstRank;
    if (relaybuffsForThread && localIndex >= 0 && localIndex < relaybuffsCount) {
      relaybuff = relaybuffsForThread[localIndex];
    }
    if (relaybuff == nullptr) {
      if (is_main_thread) {
        fprintf(stderr, "AlltoAllv: relay buffer requested but not allocated.\n");
      }
      return testInvalidUsage;
    }
    relayCount = count * relayBufferSizeFactor;
  }

  ncclResult_t res = ncclAlltoAllv(sptr, planPtr->sendcounts.data(), planPtr->sdispls.data(),
                                  rptr, planPtr->recvcounts.data(), planPtr->rdispls.data(),
                                  relaybuff, relayCount, type, comm, stream);
  if (res != ncclSuccess) {
    if (res == ncclInvalidArgument && is_main_thread) {
      fprintf(stderr,
              "AlltoAllv: ncclAlltoAllv returned invalid argument. "
              "Ensure send/recv buffers are registered as symmetric windows (use -R 2).\n");
    }
    return testNcclError;
  }
  return testSuccess;
}

struct testColl alltoAllvTest = {
  "AlltoAllv",
  AlltoAllvGetCollByteCount,
  AlltoAllvInitData,
  AlltoAllvGetBw,
  AlltoAllvRunColl
};

void AlltoAllvGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AlltoAllvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, /*eltSize=*/1, nranks);
}

testResult_t AlltoAllvRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  (void)op;
  (void)opName;
  const int firstRank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus;
  relaybuffsForThread = args->relaybuffs;
  relaybuffsFirstRank = firstRank;
  relaybuffsCount = args->nGpus;

  args->collTest = &alltoAllvTest;
  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;

  if ((int)type != -1) {
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else {
    type_count = test_typenum;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  for (int i = 0; i < type_count; i++) {
    TESTCHECK(TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "none", -1));
  }
  return testSuccess;
}

struct testEngine alltoAllvEngine = {
  .getBuffSize = AlltoAllvGetBuffSize,
  .runTest = AlltoAllvRunTest
};

#pragma weak ncclTestEngine=alltoAllvEngine
