# Build an AI Agent From Scratch in Python

Code for the Medium article **[Build AI Agent From Scratch in Python](https://jupyter2607.medium.com/build-ai-agent-from-scratch)** by Ngoc.



## The core idea: the Agent Loop


```text
        task                ┌──────────────────────┐
 Human ───────────────────► │ 1. Construct prompt   │ ◄─────────┐
   ▲                        │    (goal + memory     │           │
   │ task                   │     + tool schemas)   │           │
   │ completed              └───────────┬───────────┘           │
   │                                    │                       │
┌──┴───────────┐                        ▼                       │
│  Terminate   │            ┌──────────────────────┐            │
│ · stop crit. │            │ 2. Generate response │            │
│ · return     │            │    (LLM picks a tool)│            │ observation
│   result     │            └───────────┬──────────┘            │ added to
└──────────────┘                        │                       │ memory
   ▲                                    ▼                       │
   │ terminal tool          ┌──────────────────────┐            │
   │                        │ 3. Parse action      │            │
   │                        │    (name + args)     │            │
   │                        └───────────┬──────────┘            │
   │                                    │                       │
   │                                    ▼                       │
   │                        ┌──────────────────────┐   result   │
   └────────────────────────┤ 4. Execute action    ├────────────┘
                            │    in Environment    │
                            │    (DBs · APIs · fns)│
                            └──────────────────────┘
```

The loop observes state, picks an action, runs it in the Environment, feeds the result back into memory, and repeats until a terminal tool ends it. That's the fundamental pattern under every modern agent framework — this repo just builds it in the open.

