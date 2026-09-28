# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: workflows.spec.ts >> NW-T2-05: inspects sequence tensors from Builder and overview, then deletes each session
- Location: e2e\workflows.spec.ts:1190:1

# Error details

```
Test timeout of 180000ms exceeded.
```

```
Error: apiRequestContext.delete: Target page, context or browser has been closed
```

# Page snapshot

```yaml
- generic [ref=e4]:
  - banner [ref=e5]:
    - link "NeuroWeave models" [ref=e6] [cursor=pointer]:
      - /url: /models
      - generic [ref=e20]:
        - strong [ref=e21]: NeuroWeave
        - generic [ref=e22]: MODEL STUDIO
    - navigation "Primary navigation" [ref=e23]:
      - link "Models" [ref=e24] [cursor=pointer]:
        - /url: /models
      - link "Model Builder" [ref=e28] [cursor=pointer]:
        - /url: /builder
      - link "Nodes" [ref=e33] [cursor=pointer]:
        - /url: /nodes
      - link "Datasets" [ref=e37] [cursor=pointer]:
        - /url: /datasets
      - link "Training" [ref=e40] [cursor=pointer]:
        - /url: /training
      - link "Metrics" [ref=e53] [cursor=pointer]:
        - /url: /metrics
      - link "Environment" [ref=e56] [cursor=pointer]:
        - /url: /environment
      - link "Settings" [ref=e60] [cursor=pointer]:
        - /url: /settings
  - main [ref=e64]:
    - generic [ref=e66]:
      - generic [ref=e67]:
        - generic [ref=e68]:
          - generic [ref=e69]: MODEL BUILDER
          - generic [ref=e70]: Saved
        - generic [ref=e76]:
          - textbox "Model name" [ref=e77]:
            - /placeholder: Untitled model
            - text: Tensor inspection 1790583790828
          - button "Description" [ref=e79] [cursor=pointer]
        - generic [ref=e85]:
          - generic [ref=e86]: Backend
          - combobox "Execution backend" [ref=e89] [cursor=pointer]:
            - generic [ref=e90]: PyTorch
        - generic "Optional Keras API layer over the selected native backend." [ref=e93]:
          - generic [ref=e94]: Enable Keras
          - checkbox "Enable Keras" [checked] [ref=e95]
          - strong [ref=e96]: "On"
        - generic [ref=e97]:
          - button "Undo" [disabled] [ref=e98]
          - button "Redo" [disabled] [ref=e102]
        - generic [ref=e106]:
          - button "Validate" [ref=e107] [cursor=pointer]
          - button "Save model" [ref=e114] [cursor=pointer]
          - button "Open workspace" [ref=e120] [cursor=pointer]
      - generic [ref=e124]:
        - complementary "Node catalog" [ref=e127]:
          - generic [ref=e128]:
            - generic [ref=e129]:
              - text: BUILDING BLOCKS
              - heading "Node catalog" [level=2] [ref=e130]
            - generic [ref=e131]: 149 available
          - textbox "Search nodes" [ref=e137]:
            - /placeholder: Search layers, operations, templates
          - button "Filters" [ref=e139] [cursor=pointer]
          - generic [ref=e140]:
            - button "Convolutional 18 nodes" [ref=e142] [cursor=pointer]:
              - generic [ref=e143]:
                - strong [ref=e144]: Convolutional
                - generic [ref=e145]: 18 nodes
              - generic [ref=e146]: +
            - button "Core 14 nodes" [ref=e148] [cursor=pointer]:
              - generic [ref=e149]:
                - strong [ref=e150]: Core
                - generic [ref=e151]: 14 nodes
              - generic [ref=e152]: +
            - button "Inputs 8 nodes" [ref=e154] [cursor=pointer]:
              - generic [ref=e155]:
                - strong [ref=e156]: Inputs
                - generic [ref=e157]: 8 nodes
              - generic [ref=e158]: +
            - button "Mathematical 73 nodes" [ref=e160] [cursor=pointer]:
              - generic [ref=e161]:
                - strong [ref=e162]: Mathematical
                - generic [ref=e163]: 73 nodes
              - generic [ref=e164]: +
            - button "Merge 14 nodes" [ref=e166] [cursor=pointer]:
              - generic [ref=e167]:
                - strong [ref=e168]: Merge
                - generic [ref=e169]: 14 nodes
              - generic [ref=e170]: +
            - button "Outputs 7 nodes" [ref=e172] [cursor=pointer]:
              - generic [ref=e173]:
                - strong [ref=e174]: Outputs
                - generic [ref=e175]: 7 nodes
              - generic [ref=e176]: +
            - button "Recurrent 5 nodes" [ref=e178] [cursor=pointer]:
              - generic [ref=e179]:
                - strong [ref=e180]: Recurrent
                - generic [ref=e181]: 5 nodes
              - generic [ref=e182]: +
            - button "Transformer 10 nodes" [ref=e184] [cursor=pointer]:
              - generic [ref=e185]:
                - strong [ref=e186]: Transformer
                - generic [ref=e187]: 10 nodes
              - generic [ref=e188]: +
        - generic [ref=e189]:
          - navigation "Open models" [ref=e190]:
            - tablist "Model editor tabs" [ref=e191]:
              - generic [ref=e192]:
                - tab "○ Tensor inspection 1790583790828 r1" [selected] [ref=e193] [cursor=pointer]:
                  - generic [ref=e194]: ○
                  - generic [ref=e195]: Tensor inspection 1790583790828
                  - generic [ref=e196]: r1
                - button "Close Tensor inspection 1790583790828" [ref=e197] [cursor=pointer]
            - button "New model" [ref=e201] [cursor=pointer]
          - region "Model graph canvas" [ref=e205]:
            - generic [ref=e206]:
              - generic:
                - generic:
                  - group "Connection from inspection-sequence-input|output|output to inspection-rnn|input|input"
                  - group "Connection from inspection-rnn|output|output to inspection-sequence-output|input|input"
                - generic:
                  - group "Node Sequence input" [ref=e215]:
                    - generic [ref=e216]:
                      - generic [ref=e217]: inputs
                      - strong [ref=e219]: Sequence input
                      - generic [ref=e220]: input.sequence
                      - generic [ref=e221]: "? × 3 × 4"
                      - button "Port Output (output) on Sequence input" [ref=e222]
                  - group "Node Sequence output" [ref=e224]:
                    - generic [ref=e225]:
                      - generic "Inferred workflow output" [ref=e226]
                      - generic [ref=e227]: outputs
                      - strong [ref=e229]: Sequence output
                      - generic [ref=e230]: output.sequence
                      - generic [ref=e231]: "? × 3 × 4"
                      - button "Port Input (input) on Sequence output" [ref=e232]
                      - button "Port Output (output) on Sequence output" [ref=e234]
                  - group "Node Sequence encoder" [ref=e236]:
                    - generic [ref=e237]:
                      - generic "Inferred workflow output" [ref=e238]
                      - generic [ref=e239]: recurrent
                      - strong [ref=e241]: Sequence encoder
                      - generic [ref=e242]: recurrent.simple_rnn
                      - generic [ref=e243]: "? × 3 × 4"
                      - button "Port Input (input) on Sequence encoder" [ref=e244]
                      - generic:
                        - button "Port Output (output) on Sequence encoder" [ref=e246]
                        - button "Port Hidden state (output) on Sequence encoder" [ref=e248]
              - generic [ref=e250]: MINI MAP
              - group "Canvas zoom controls" [ref=e257]:
                - button "Zoom in" [ref=e258] [cursor=pointer]: +
                - button "Zoom out" [ref=e259] [cursor=pointer]: −
                - button "Reset view" [ref=e260] [cursor=pointer]
              - toolbar "Graph editor controls" [ref=e266]:
                - generic [ref=e267]:
                  - button "Hide minimap" [ref=e268] [cursor=pointer]: Hide map
                  - button "Lock canvas" [ref=e269] [cursor=pointer]
                - button "Enter full canvas" [ref=e274] [cursor=pointer]: Focus
            - generic: Drop a node to build the model • Ctrl+L auto-layout
          - button "GRAPH HEALTH Validation No findings Ready" [ref=e280] [cursor=pointer]:
            - generic [ref=e281]:
              - generic [ref=e282]: GRAPH HEALTH
              - strong [ref=e283]: Validation
              - generic [ref=e284]: No findings
            - generic [ref=e285]: Ready
        - complementary "Properties inspector" [ref=e294]:
          - generic [ref=e295]:
            - generic [ref=e296]:
              - generic [ref=e297]: recurrent · NODE
              - heading "Sequence encoder" [level=2] [ref=e298]
              - code [ref=e299]: recurrent.simple_rnn
            - generic [ref=e300]: Ready
          - generic [ref=e302]:
            - generic [ref=e303]:
              - text: Label
              - textbox "Label" [ref=e304]: Sequence encoder
            - paragraph [ref=e305]: Simple RNN operation.
            - group [ref=e306]:
              - generic "Advanced 3" [ref=e307] [cursor=pointer]:
                - text: Advanced
                - generic [ref=e308]: "3"
              - option "None" [selected]
              - option "float16"
              - option "float32"
              - option "float64"
              - option "bfloat16"
              - option "int8"
              - option "int16"
              - option "int32"
              - option "int64"
              - option "uint8"
              - option "uint16"
              - option "bool"
            - group [ref=e309]:
              - generic "Activation 2" [ref=e310] [cursor=pointer]:
                - text: Activation
                - generic [ref=e311]: "2"
            - group [ref=e312]:
              - generic "General 1" [ref=e313] [cursor=pointer]:
                - text: General
                - generic [ref=e314]: "1"
              - generic [ref=e315]:
                - generic [ref=e316]: Use bias
                - generic [ref=e318]:
                  - generic [ref=e319]: editable
                  - generic [ref=e320]:
                    - checkbox "Use bias editable Enabled" [checked] [ref=e321]
                    - generic [ref=e322]: Enabled
            - group [ref=e323]:
              - generic "Regularization 6" [ref=e324] [cursor=pointer]:
                - text: Regularization
                - generic [ref=e325]: "6"
              - option "none" [selected]
              - option "l1"
              - option "l2"
              - option "l1_l2"
              - option "none" [selected]
              - option "l1"
              - option "l2"
              - option "l1_l2"
              - option "none" [selected]
              - option "l1"
              - option "l2"
              - option "l1_l2"
              - option "none" [selected]
              - option "l1"
              - option "l2"
              - option "l1_l2"
            - group [ref=e326]:
              - generic "Behavior 6" [ref=e327] [cursor=pointer]:
                - text: Behavior
                - generic [ref=e328]: "6"
              - generic [ref=e329]:
                - generic [ref=e330]: Return sequences
                - generic [ref=e332]:
                  - generic [ref=e333]: editable
                  - generic [ref=e334]:
                    - checkbox "Return sequences editable Enabled" [checked] [ref=e335]
                    - generic [ref=e336]: Enabled
              - generic [ref=e337]:
                - generic [ref=e338]: Return state
                - generic [ref=e340]:
                  - generic [ref=e341]: editable
                  - generic [ref=e342]:
                    - checkbox "Return state editable Enabled" [checked] [ref=e343]
                    - generic [ref=e344]: Enabled
              - generic [ref=e345]:
                - generic [ref=e346]: Go backwards
                - generic [ref=e348]:
                  - generic [ref=e349]: editable
                  - generic [ref=e350]:
                    - checkbox "Go backwards editable Disabled" [ref=e351]
                    - generic [ref=e352]: Disabled
              - generic [ref=e353]:
                - generic [ref=e354]: Stateful
                - generic [ref=e356]:
                  - generic [ref=e357]: editable
                  - generic [ref=e358]:
                    - checkbox "Stateful editable Disabled" [ref=e359]
                    - generic [ref=e360]: Disabled
              - generic [ref=e361]:
                - generic [ref=e362]: Unroll
                - generic [ref=e364]:
                  - generic [ref=e365]: editable
                  - generic [ref=e366]:
                    - checkbox "Unroll editable Disabled" [ref=e367]
                    - generic [ref=e368]: Disabled
              - generic [ref=e369]:
                - generic [ref=e370]: Seed
                - generic [ref=e372]:
                  - generic [ref=e373]: editable
                  - spinbutton "Seed editable" [ref=e374]
            - group [ref=e375]:
              - generic "Shape 1" [ref=e376] [cursor=pointer]:
                - text: Shape
                - generic [ref=e377]: "1"
              - generic [ref=e378]:
                - generic [ref=e379]: Units
                - generic [ref=e381]:
                  - generic [ref=e382]: editable
                  - spinbutton "Units editable" [ref=e383]: "4"
            - group [ref=e384]:
              - generic "Initializers 3" [ref=e385] [cursor=pointer]:
                - text: Initializers
                - generic [ref=e386]: "3"
              - option "uniform"
              - option "normal"
              - option "truncated_normal"
              - option "constant"
              - option "variance_scaling"
              - option "glorot_uniform" [selected]
              - option "glorot_normal"
              - option "he_uniform"
              - option "he_normal"
              - option "lecun_uniform"
              - option "lecun_normal"
              - option "zeros"
              - option "ones"
              - option "random_normal"
              - option "random_uniform"
              - option "orthogonal"
              - option "identity"
              - option "uniform"
              - option "normal"
              - option "truncated_normal"
              - option "constant"
              - option "variance_scaling"
              - option "glorot_uniform"
              - option "glorot_normal"
              - option "he_uniform"
              - option "he_normal"
              - option "lecun_uniform"
              - option "lecun_normal"
              - option "zeros"
              - option "ones"
              - option "random_normal"
              - option "random_uniform"
              - option "orthogonal" [selected]
              - option "identity"
              - option "uniform"
              - option "normal"
              - option "truncated_normal"
              - option "constant"
              - option "variance_scaling"
              - option "glorot_uniform"
              - option "glorot_normal"
              - option "he_uniform"
              - option "he_normal"
              - option "lecun_uniform"
              - option "lecun_normal"
              - option "zeros" [selected]
              - option "ones"
              - option "random_normal"
              - option "random_uniform"
              - option "orthogonal"
              - option "identity"
            - group [ref=e387]:
              - generic "Constraints 3" [ref=e388] [cursor=pointer]:
                - text: Constraints
                - generic [ref=e389]: "3"
              - option "none" [selected]
              - option "max_norm"
              - option "min_max_norm"
              - option "non_neg"
              - option "unit_norm"
              - option "none" [selected]
              - option "max_norm"
              - option "min_max_norm"
              - option "non_neg"
              - option "unit_norm"
              - option "none" [selected]
              - option "max_norm"
              - option "min_max_norm"
              - option "non_neg"
              - option "unit_norm"
            - region "CONNECTIONS" [ref=e390]:
              - generic [ref=e391]:
                - generic [ref=e392]:
                  - generic [ref=e393]: CONNECTIONS
                  - generic [ref=e394]: Data entering and leaving this node
                - generic [ref=e395]: "3"
              - generic [ref=e396]:
                - generic [ref=e397]:
                  - generic [ref=e398]: INPUT
                  - generic [ref=e399]:
                    - strong [ref=e400]: Input
                    - generic [ref=e401]: tensor · shape [?, 3, 4] · float32
                    - generic [ref=e402]:
                      - generic [ref=e403]: Connected
                      - generic [ref=e404]: ·Required
                      - generic [ref=e405]: ·One connection
                - generic [ref=e406]:
                  - generic [ref=e407]: OUTPUT
                  - generic [ref=e408]:
                    - strong [ref=e409]: Output
                    - generic [ref=e410]: tensor · shape [?, 3, 4] · float32
                    - generic [ref=e411]:
                      - generic [ref=e412]: Connected
                      - generic [ref=e413]: ·Required
                      - generic [ref=e414]: ·Fan-out allowed
                      - button "Use Output as a model output" [disabled] [ref=e415] [cursor=pointer]: Use as output
                - generic [ref=e416]:
                  - generic [ref=e417]: OUTPUT
                  - generic [ref=e418]:
                    - strong [ref=e419]: Hidden state
                    - generic [ref=e420]: tensor · shape [?, 4] · float32
                    - generic [ref=e421]:
                      - generic [ref=e422]: Not connected
                      - generic [ref=e423]: ·Optional
                      - generic [ref=e424]: ·Fan-out allowed
                      - button "Use Hidden state as a model output" [disabled] [ref=e425] [cursor=pointer]: Use as output
      - button "Model graph is valid." [ref=e426] [cursor=pointer]
  - dialog [ref=e428]:
    - banner [ref=e429]:
      - generic [ref=e430]:
        - text: TENSOR INSPECTION
        - heading "Preparing inspection…" [level=2] [ref=e431]
        - paragraph [ref=e432]: Instrumented forward pass
      - button "Close tensor inspection" [active] [ref=e433] [cursor=pointer]
    - alert [ref=e437]: Device 'cpu' already has an active ML job
```

# Test source

```ts
  68  |         position: { x: 960, y: 144 },
  69  |         parameters: {},
  70  |       },
  71  |     ],
  72  |     edges: [
  73  |       { id: 'edge-e2e-1', source_node_id: 'input-tabular-e2e', source_port_id: 'output', target_node_id: 'layer-dense-e2e-1', target_port_id: 'input' },
  74  |       { id: 'edge-e2e-2', source_node_id: 'layer-dense-e2e-1', source_port_id: 'output', target_node_id: 'layer-dense-e2e-2', target_port_id: 'input' },
  75  |       { id: 'edge-e2e-3', source_node_id: 'layer-dense-e2e-2', source_port_id: 'output', target_node_id: 'output-classification-e2e', target_port_id: 'input' },
  76  |     ],
  77  |     model_configuration: {
  78  |       keras: {
  79  |         optimizer: 'adam',
  80  |         optimizer_parameters: { learning_rate: 0.001 },
  81  |         losses: { __default__: 'binary_crossentropy' },
  82  |         metrics: { __default__: ['binary_accuracy', 'auc'] },
  83  |         default_initializer: 'glorot_uniform',
  84  |         default_regularizer: 'none',
  85  |         default_constraint: 'none',
  86  |         mixed_precision: false,
  87  |         random_seed: 42,
  88  |       },
  89  |       native: {
  90  |         tensorflow: { jit_compile: false },
  91  |         pytorch: {
  92  |           optimizer: 'adamw',
  93  |           optimizer_parameters: { learning_rate: 0.0005, weight_decay: 0.004, beta_1: 0.9, beta_2: 0.999, epsilon: 1e-7 },
  94  |           losses: { __default__: 'binary_crossentropy' },
  95  |           metrics: { __default__: ['binary_accuracy', 'auc'] },
  96  |           compile_enabled: false,
  97  |           mode: 'default',
  98  |           fullgraph: false,
  99  |           dynamic: false,
  100 |         },
  101 |         jax: { jit_enabled: false },
  102 |       },
  103 |     },
  104 |     export_settings: { include_comments: true, include_shape_assertions: true, preferred_binary_format: 'keras' },
  105 |     future_training_configuration: {},
  106 |   }
  107 | }
  108 |
  109 | function sequenceInspectionGraph() {
  110 |   const base = graph()
  111 |   return {
  112 |     ...base,
  113 |     nodes: [
  114 |       { ...base.nodes[0], id: 'inspection-sequence-input', definition_id: 'input.sequence', label: 'Sequence input', position: { x: 48, y: 144 }, parameters: { shape: [3, 4], dtype: 'float32' } },
  115 |       { ...base.nodes[1], id: 'inspection-rnn', definition_id: 'recurrent.simple_rnn', label: 'Sequence encoder', position: { x: 400, y: 144 }, parameters: { units: 4, return_sequences: true, return_state: true } },
  116 |       { ...base.nodes[3], id: 'inspection-sequence-output', definition_id: 'output.sequence', label: 'Sequence output', position: { x: 752, y: 144 }, parameters: {} },
  117 |     ],
  118 |     edges: [
  119 |       { id: 'inspection-sequence-edge-1', source_node_id: 'inspection-sequence-input', source_port_id: 'output', target_node_id: 'inspection-rnn', target_port_id: 'input' },
  120 |       { id: 'inspection-sequence-edge-2', source_node_id: 'inspection-rnn', source_port_id: 'output', target_node_id: 'inspection-sequence-output', target_port_id: 'input' },
  121 |     ],
  122 |   }
  123 | }
  124 |
  125 | function fanOutGraph() {
  126 |   const base = graph()
  127 |   const denseParameters = (units: number, activation: string) => ({
  128 |     units,
  129 |     activation,
  130 |     use_bias: true,
  131 |     kernel_initializer: 'glorot_uniform',
  132 |     bias_initializer: 'zeros',
  133 |     kernel_regularizer: 'none',
  134 |     bias_regularizer: 'none',
  135 |     activity_regularizer: 'none',
  136 |     kernel_constraint: 'none',
  137 |     bias_constraint: 'none',
  138 |   })
  139 |   return {
  140 |     ...base,
  141 |     nodes: [
  142 |       { ...base.nodes[0], id: 'input-branch-e2e', label: 'Branch source', position: { x: 48, y: 160 } },
  143 |       { ...base.nodes[1], id: 'dense-primary-e2e', label: 'Primary branch', position: { x: 352, y: 80 }, parameters: denseParameters(16, 'relu') },
  144 |       { ...base.nodes[2], id: 'merge-add-e2e', definition_id: 'merge.add', label: 'Merge', position: { x: 656, y: 160 }, parameters: {} },
  145 |       { ...base.nodes[1], id: 'dense-downstream-e2e', label: 'Downstream', position: { x: 960, y: 80 }, parameters: denseParameters(1, 'sigmoid') },
  146 |       { ...base.nodes[3], id: 'output-branch-e2e', label: 'Prediction', position: { x: 1264, y: 80 } },
  147 |       { ...base.nodes[1], id: 'dense-fanout-e2e', label: 'Fan-out target', position: { x: 352, y: 360 }, parameters: denseParameters(8, 'tanh') },
  148 |       { ...base.nodes[2], id: 'mask-target-e2e', definition_id: 'transformer.attention_mask', label: 'Mask target', position: { x: 656, y: 440 }, parameters: {} },
  149 |     ],
  150 |     edges: [
  151 |       { id: 'edge-branch-1', source_node_id: 'input-branch-e2e', source_port_id: 'output', target_node_id: 'dense-primary-e2e', target_port_id: 'input' },
  152 |       { id: 'edge-branch-2', source_node_id: 'dense-primary-e2e', source_port_id: 'output', target_node_id: 'merge-add-e2e', target_port_id: 'inputs' },
  153 |       { id: 'edge-branch-3', source_node_id: 'merge-add-e2e', source_port_id: 'output', target_node_id: 'dense-downstream-e2e', target_port_id: 'input' },
  154 |       { id: 'edge-branch-4', source_node_id: 'dense-downstream-e2e', source_port_id: 'output', target_node_id: 'output-branch-e2e', target_port_id: 'input' },
  155 |     ],
  156 |   }
  157 | }
  158 |
  159 | async function createProject(request: APIRequestContext, name: string, modelGraph = graph()): Promise<CreatedProject> {
  160 |   const response = await request.post('/api/projects', {
  161 |     data: { name, description: 'Browser regression fixture', execution_backend: 'pytorch', keras_enabled: true, graph: modelGraph },
  162 |   })
  163 |   expect(response.ok()).toBeTruthy()
  164 |   return response.json() as Promise<CreatedProject>
  165 | }
  166 |
  167 | async function deleteProject(request: APIRequestContext, id: string): Promise<void> {
> 168 |   const response = await request.delete(`/api/projects/${id}`)
      |                                        ^ Error: apiRequestContext.delete: Target page, context or browser has been closed
  169 |   expect([204, 404]).toContain(response.status())
  170 | }
  171 |
  172 | async function dragConnection(page: import('@playwright/test').Page, sourceId: string, targetId: string): Promise<void> {
  173 |   const source = page.locator(`[data-f-connector-id="${sourceId}"]`)
  174 |   const target = page.locator(`[data-f-connector-id="${targetId}"]`)
  175 |   const sourceBounds = await source.boundingBox()
  176 |   const targetBounds = await target.boundingBox()
  177 |   expect(sourceBounds).not.toBeNull()
  178 |   expect(targetBounds).not.toBeNull()
  179 |   await page.mouse.move(sourceBounds!.x + sourceBounds!.width / 2, sourceBounds!.y + sourceBounds!.height / 2)
  180 |   await page.mouse.down()
  181 |   await page.mouse.move(targetBounds!.x + targetBounds!.width / 2, targetBounds!.y + targetBounds!.height / 2, { steps: 8 })
  182 | }
  183 |
  184 | function backendCompatibilityGraph() {
  185 |   const base = graph()
  186 |   return {
  187 |     ...base,
  188 |     nodes: [
  189 |       { ...base.nodes[0], id: 'input-sequence-compat-e2e', definition_id: 'input.sequence', label: 'Sequence input', position: { x: 48, y: 160 }, parameters: { shape: [4, 3], dtype: 'float32' } },
  190 |       { ...base.nodes[1], id: 'simple-rnn-compat-e2e', definition_id: 'recurrent.simple_rnn', label: 'Simple RNN', position: { x: 400, y: 160 }, parameters: { units: 8, return_sequences: true, return_state: true } },
  191 |       { ...base.nodes[3], id: 'output-sequence-compat-e2e', definition_id: 'output.sequence', label: 'Sequence output', position: { x: 752, y: 160 }, parameters: {} },
  192 |     ],
  193 |     edges: [
  194 |       { id: 'edge-compat-e2e-1', source_node_id: 'input-sequence-compat-e2e', source_port_id: 'output', target_node_id: 'simple-rnn-compat-e2e', target_port_id: 'input' },
  195 |       { id: 'edge-compat-e2e-2', source_node_id: 'simple-rnn-compat-e2e', source_port_id: 'output', target_node_id: 'output-sequence-compat-e2e', target_port_id: 'input' },
  196 |     ],
  197 |   }
  198 | }
  199 |
  200 | async function waitForCanvasFit(page: import('@playwright/test').Page): Promise<void> {
  201 |   await expect.poll(async () => {
  202 |     const style = await page.locator('f-canvas').getAttribute('style')
  203 |     return Boolean(style && !style.includes('matrix(1, 0, 0, 1, 0, 0)'))
  204 |   }, { timeout: 5_000 }).toBeTruthy()
  205 | }
  206 |
  207 | test('imports a Keras architecture through the browser file input', async ({ page, request }) => {
  208 |   test.setTimeout(90_000)
  209 |   let projectId: string | undefined
  210 |   try {
  211 |     await page.goto('/environment')
  212 |     const pytorchCard = page.locator('.backend-status-card-container').filter({ hasText: 'PyTorch' })
  213 |     const kerasRuntimeReady = async () => {
  214 |       const metadataResponse = await request.get('/api/metadata')
  215 |       const metadata = await metadataResponse.json() as {
  216 |         execution_backends: Array<{
  217 |           id: string
  218 |           runtime_apis?: Array<{ id: string; available: boolean; state?: string }>
  219 |         }>
  220 |       }
  221 |       const keras = metadata.execution_backends.find((backend) => backend.id === 'pytorch')?.runtime_apis?.find((api) => api.id === 'keras')
  222 |       return Boolean(keras?.available && keras.state === 'verified')
  223 |     }
  224 |     if (!(await kerasRuntimeReady())) {
  225 |       const kerasCheck = pytorchCard.getByRole('button', { name: 'Check Keras', exact: true })
  226 |       await expect(kerasCheck).toBeVisible({ timeout: 15_000 })
  227 |       await kerasCheck.click()
  228 |     }
  229 |     await expect.poll(kerasRuntimeReady, { timeout: 90_000 }).toBeTruthy()
  230 |     await expect(pytorchCard).toContainText('Usable')
  231 |
  232 |     await page.goto('/models')
  233 |     const importResponse = page.waitForResponse((response) => response.url().includes('/api/imports/keras-architecture') && response.request().method() === 'POST')
  234 |     await page.locator('input[type="file"][accept*=".keras"]').setInputFiles({
  235 |       name: 'browser-import.keras',
  236 |       mimeType: 'application/vnd.keras.zip',
  237 |       buffer: kerasImportFixture,
  238 |     })
  239 |     const response = await importResponse
  240 |     expect(response.status()).toBe(201)
  241 |     const imported = await response.json() as { project: { id: string; name: string; graph: { nodes: unknown[] } } }
  242 |     projectId = imported.project.id
  243 |
  244 |     await expect(page).toHaveURL(new RegExp(`/models/${projectId}$`))
  245 |     await expect(page.getByRole('heading', { name: 'browser import' })).toBeVisible()
  246 |     expect(imported.project.graph.nodes.length).toBe(4)
  247 |     const persisted = await (await request.get(`/api/projects/${projectId}`)).json() as { name: string; graph: { nodes: unknown[] } }
  248 |     expect(persisted.name).toBe('browser import')
  249 |     expect(persisted.graph.nodes).toHaveLength(4)
  250 |   } finally {
  251 |     if (projectId) await deleteProject(request, projectId)
  252 |   }
  253 | })
  254 |
  255 | test('NW-T1-03 exports and reimports a project bundle through the browser', async ({ page, request }) => {
  256 |   const source = await createProject(request, `NW-T1-03 bundle ${Date.now()}`)
  257 |   let importedId: string | undefined
  258 |   try {
  259 |     const sourceResponse = await request.get(`/api/projects/${source.id}`)
  260 |     expect(sourceResponse.ok()).toBeTruthy()
  261 |     const original = await sourceResponse.json() as {
  262 |       name: string
  263 |       execution_backend: string
  264 |       keras_enabled: boolean
  265 |       graph: unknown
  266 |     }
  267 |
  268 |     await page.goto(`/models/${source.id}/export`)
```