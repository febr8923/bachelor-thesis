### Step-by-step setup:
- Select node
    - Show nodes: sinfo -N -l -e or cat /etc/motd
    - Salloc to select node: salloc -t 04:00:00 -w ault<number> (Use only UV gpus)
    - Select node: srun -i --pty bash or srun --pty -n 1 bash -i 
    - (Check if on node with hostname)
- Create second terminal (tmux, then ctrl + b, %)
    -  echo $SLURM_JOB_ID
    -  export SLURM_JOB_ID=<job-id-of-your-job>
    -  srun --pty --oversubscribe -n 1 --pty bash -i
- Load modules
    -  module load cuda/11.6.2
    -  module load gcc/10.2
- Set path variables
    -  export REPO_DIR="/users/fbrunne/projects/mignificient"
    -  export OPENSSL_PATH="/users/fbrunne/anaconda3/pkgs/openssl-3.4.1- h7b32b05_0"
    -  export DEPS_PATH="/users/fbrunne/anaconda3/pkgs/libacl-2.3.2-h0f662aa_0"
    -  export BUILD_DIR="/users/fbrunne/projects/mignificient_build“ 
- Apply iceoryx patch (once)
    -  git apply ../iceoryx.patch
    - Executed in external/iceoryx
- CMAKE project (in build directory):
    -  jsoncpp_DIR=/users/fbrunne/anaconda3/lib/cmake/jsoncpp pybind11_DIR=/users/fbrunne/anaconda3/lib/python3.12/site-packages/pybind11 cmake -DCUDNN_DIR=/users/fbrunne/anaconda3/pkgs/cudnn-8.1.0.77-h90431f1_0/ -DCMAKE_C_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_FLAGS="-I ${DEPS_PATH}/include" -DCMAKE_CXX_STANDARD_LIBRARIES="-L${DEPS_PATH}/lib" -DCMAKE_PREFIX_PATH=/scratch/mcopik/gpus/deps -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="70;80" -DOPENSSL_ROOT_DIR=${OPENSSL_PATH} -DOPENSSL_LIBRARIES=${OPENSSL_PATH}/lib -B . -S /users/fbrunne/projects/mignificient
    - To compile with info: -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO
    - 
- Make the project
    -   make- j
- Generate device config:
    -  ${REPO_DIR}/tools/list-gpus.sh logs_$(hostname)
    - Executed in REPO_DIR
- Start iceoryx:
    -  ${REPO_DIR}/tools/start.sh ${BUILD_DIR} logs_$(hostname)
    - Executed in REPO_DIR
- Start orchestrator:
    -  ${BUILD_DIR}/orchestrator/orchestrator ${BUILD_DIR}/config/orchestrator.json ${REPO_DIR}/logs_$(hostname)/devices.json > orchestrator_output.log 2>&1 &
- Start invoker:
    -  ${BUILD_DIR}/invoker/bin/invoker ${BUILD_DIR}/examples/vector_add.json result.csv
    - Needs to be executed in the same directory the orchestrator was started!

When changing code:
- Set path variables (once)
    -  export REPO_DIR="/users/fbrunne/projects/mignificient"
    -  export BUILD_DIR="/users/fbrunne/projects/mignificient_build“
- Make the project
    -  make -j


### mignificient overview:

Orchestrator receives a request
Orchestrator spawns a new gpuless server and function (client)
Processes are running
Client starts function execution
CUDA init (before + after)
Client stops execution

Measure cost of starting executor & everything else
- Start user & user executor 


Executor/src/executor_cpp.cpp
- While loop: processing requests
- Func: start executor

Two components: 
- One injec
- Server (that we spawn in orchestrator)
    - gputless/src .. manager/manager_device_cli.cpp
    - Implements functions (manage_device) from file before; Find it at gputless/src .. manager/manager_device.cpp
        - Getcudavirtualdevice: initializes the cuda context (once), measure (start timestamp)
    - 


The orchestrator handles invocations at the /invoke HTTP endpoint.
1. HTTP Request Reception:
    * The HTTPServer::invoke method in http.cpp receives the POST request at /invoke.
    * It extracts the request body, which is expected to be a JSON payload.
2. Invocation Object Creation:
    * An ActiveInvocation object is created from the JSON payload. This object encapsulates all the details of the invocation, such as function name, user, input data, required GPU memory, etc. (defined in invocation.hpp).
    * If parsing the JSON or validating required fields fails, an HTTP error response is sent back immediately.
3. Triggering the Orchestrator:
    * If the ActiveInvocation object is created successfully, the HTTPTrigger::trigger method is called.
    * This method adds the ActiveInvocation to a queue and signals the main orchestrator event loop using an iceoryx_posh::popo::UserTrigger.
4. Orchestrator Event Loop Processing:
    * The main Orchestrator::event_loop (in orchestrator.cpp) is waiting for events.
    * When the UserTrigger is fired, the Orchestrator::_handle_http static method is called.
    * _handle_http retrieves all pending ActiveInvocation objects from the HTTPTrigger's queue.
5. Processing Each Invocation:
    * For each ActiveInvocation, the Users::process_invocation method (in users.hpp) is called. This is the core logic for deciding how and where to run the function.
    * Users::process_invocation attempts to find or allocate a suitable Client (representing an execution environment, potentially a container or a bare-metal process) and a GPUInstance.
    * The logic involves:
        * Checking for existing idle clients for the same user and function.
        * If not found, checking for idle GPUs to allocate a new client.
        * If no idle GPUs, finding the least busy GPU.
        * If a new client needs to be created, the Users::allocate method is called. This involves:
            * Creating a Client object.
            * Potentially starting a new executor process (e.g., BareMetalExecutorPython::start or SarusContainerExecutorCpp::start in executor.cpp).
            * Setting up communication channels (iceoryx publishers/subscribers) between the orchestrator and the new executor.
        * If no suitable GPU/client can be found (e.g., due to insufficient memory), the invocation is rejected with an HTTP error.
6. Dispatching to Client and GPU:
    * Once a Client and GPUInstance are selected/allocated:
        * The ActiveInvocation is added to the Client's pending queue (Client::add_invocation).
        * The invocation is also added to the GPUInstance's pending queue (GPUInstance::add_invocation).
        * The GPUInstance then tries to schedule_next invocation.
7. Execution and Response:
    * The Client sends the request to its associated executor process via iceoryx.
    * The GPUInstance manages the execution flow on the GPU, potentially activating memcpy or kernel execution phases for the gpuless component (Client::activate_memcpy, Client::activate_kernels).
    * When the executor finishes, it sends a result back to the orchestrator via iceoryx.
    * The handle_client function (in orchestrator.cpp) receives this result.
    * The Client::finished method is called, which in turn calls ActiveInvocation::respond.
    * ActiveInvocation::respond sends an HTTP 200 OK response back to the original caller of the /invoke endpoint, including any output from the function execution.
    * If the execution yields (Client::yield), the GPUInstance might schedule another pending invocation if the sharing model allows.