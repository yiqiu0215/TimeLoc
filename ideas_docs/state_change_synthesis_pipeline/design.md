# StateChangeRecognition 数据合成 Pipeline 设计

## 1. 目标与样本口径

目标是将现有 GEB+ 训练数据与新合成标注合并，形成约 200K--400K 个可用于“给定视频、时间戳和 Subject，预测 Status Before/After”的训练样本。

统一将一个样本定义为唯一的：

```text
(source_dataset, source_video_id, timestamp, subject)
```

同一边界的多人复述、同义改写和多次 API 采样不重复计数。项目采用的确实是 [GEB+ 官方发布](https://yuxuan-w.github.io/GEB-plus/)；其论文报告 12,434 个视频、176,681 条 boundary-caption 标注，但这是多人标注口径。项目本地约 40K 更接近去重后的 boundary-subject 样本口径，应先按上述唯一键重新统计，不能直接与论文的 176K 相加。

## 2. 数据集筛选结论

时间信息是硬门槛：只有官方标注提供 `timestamp`、`start/end seconds`、`start/end frame` 或固定时间格点的数据集才进入候选池。动作类别标签但没有内部时间位置的数据集不进入主 pipeline。

### 2.1 第一优先级：精确时间标注且容易生成候选

| 数据集 | 官方规模 | 原生时间标注 | Subject 先验 | 建议用途 |
| --- | ---: | --- | --- | --- |
| Perception Test | 11,353 个含动作段的视频、73,503 个动作段 | 动作 `start/end timestamp` 和 `start/end frame` | `parent_objects`、object track、对象名称 | 核心来源，预计保留 35K--50K |
| EPIC-KITCHENS-100 | 100 小时、约 90K 动作段 | action start/stop timestamp 和 frame | noun、verb、narration | 核心来源，预计保留 40K--60K |
| Assembly101 | 4,321 个多视角视频、100K coarse/1M fine action segments | coarse/fine action start/end | 动作 verb-object、手部和视角信息 | 核心来源；每个 take 只选一个主视角，预计保留 60K--120K |
| Ego4D FHO / OSCC-PNR | Ego4D 共 3,700+ 小时 | pre、PNR、post 帧/时间，PNR 是直接边界 | 物体框、state-change type、verb、noun | 任务匹配度最高；许可批准后预计保留 20K--40K |
| Charades | 9,848 个视频、66,500 个 temporal action annotations | action start/end seconds | action/object labels | 日常室内补充，预计保留 20K--35K |

### 2.2 第二优先级：有时间标注，但需要更多筛选或许可审查

| 数据集 | 官方规模 | 原生时间标注 | 主要限制 |
| --- | ---: | --- | --- |
| Ego-Exo4D Keysteps | V2 为 5,035 takes、1,286.3 视频小时 | 每个 keystep 有 `start_time/end_time` 和文字描述 | 多视角重复；必须接受数据协议并确认 API 处理权 |
| HACS Segments | 50K untrimmed videos、139K action segments | densely annotated segment boundaries | YouTube 来源，下载可用性和第三方 API 权利需审查 |
| COIN | 11,827 个视频、180 个任务 | 每个 procedure step 有 temporal boundaries 和描述 | YouTube 来源；边界较粗，需局部细化 |
| ActivityNet Captions | 20K 视频、约 100K temporal descriptions | 每条 caption 有 start/end seconds | 事件段较长、边界较粗；YouTube 来源 |
| YouCook2 | 2,000 个长视频、约 15K procedure segments | 每个步骤有 start/end time 和句子 | 仅烹饪，规模较小且为 YouTube 来源 |
| IKEA ASM | 371 个多视角 assembly samples | 每个 atomic action 有 start/end frame | 规模小、装配偏置、多视角重复；CC-BY-NC 4.0 |
| FineGym | 体操比赛视频及细粒度动作实例 | action/sub-action 两层 temporal bounds | 体育转播和版权风险；主体变化以人为主 |
| VidSitu | 29K 个 10 秒电影片段、145K events | 每 2 秒一个事件和语义角色 | 时间格点不是人工边界；电影内容的 API/衍生发布风险较高 |

### 2.3 排除或暂缓

- Something-Something V2：虽有 220,847 个动作短视频，但没有动作内部边界时间戳；还存在禁止向第三方提供数据的许可约束，不进入闭源 API 主方案。
- ChangeIt：任务语义相关，但缺少适合直接映射 GEB+ 的统一人工边界时间戳，不作为首批来源。
- Kinetics-400/700：现有 GEB+ 来自 Kinetics-400，继续加入会造成源视频重复和评测污染。
- HowTo100M/WebVid：没有适合直接使用的高质量边界标注，版权、失效链接和噪声问题显著。

## 3. 推荐规模规划

先以约 260K 总样本为一期目标，而不是直接冲到 400K：

| 来源 | 一期目标（通过质检后） |
| --- | ---: |
| 现有 GEB+（按唯一键重算） | 约 40K |
| Assembly101 | 70K |
| EPIC-KITCHENS-100 | 45K |
| Perception Test | 40K |
| Charades | 25K |
| 获得许可后的 Ego4D FHO | 25K |
| COIN/HACS/Ego-Exo4D 中通过权利审查的数据 | 15K |
| 合计 | 约 260K |

若一期人工审计达到预设质量门槛，再将 Assembly101 扩到 100K--120K，并加入获准的 Ego-Exo4D、HACS、ActivityNet Captions 或自采且带时间标注的视频，将总量扩至 300K--400K。不要通过同一视频多视角复制、同义改写或一个状态变化拆成多个近邻时间戳来虚增规模。

## 4. Pipeline 总体流程

```text
License Gate
  -> Raw Dataset Registry
  -> Video/Annotation Normalization
  -> Candidate Boundary Proposal
  -> Clip and Frame-Package Construction
  -> VLM Boundary Refinement + Captioning
  -> Independent VLM Verification
  -> Rule Validation and Deduplication
  -> Human Stratified Audit
  -> GEB+ JSON Export + Provenance Sidecar
```

### 4.1 License Gate

每个数据源建立 `dataset_registry.yaml`，至少记录：

```yaml
dataset_name: perception_test
license_name: CC-BY-4.0
license_url: ...
commercial_use: true
derivative_annotations: allowed
third_party_api_processing: reviewed
video_redistribution: reviewed
allowed_splits: [train]
review_date: YYYY-MM-DD
```

`third_party_api_processing` 未被明确审查为允许时，调度器不得生成 API 请求。受限数据集可以走本地 VLM pipeline，但不能通过“只传帧不传视频”规避条款，因为帧仍属于原始数据。

闭源 API 还需记录数据保留策略、训练使用策略、部署区域和文件删除策略。若使用云服务，优先选择企业/云版本的数据控制，而不是默认消费者接口。

### 4.2 原始数据注册与标准化

原视频只读保存，任何裁剪、转码和帧抽取均写到新目录。每个视频记录：

```text
source_dataset, source_split, source_video_id, local_path,
sha256, perceptual_hash, duration, fps, width, height,
license_id, participant_id/take_id, original_annotation_path
```

统一为 H.264 MP4 仅作为 API 工作副本；保留原始文件和映射。检查视频时长、损坏文件、音画时间轴和重复视频。根据 SHA-256、视觉指纹和来源 ID 与现有 GEB+ 去重。

### 4.3 候选边界生成

不要让闭源 VLM 从长视频中无约束地寻找所有边界。先利用原数据标注生成高召回候选：

- Ego4D：直接使用 PNR，pre/post 时间作为上下文锚点。
- EPIC-KITCHENS：动作段起点、终点及相邻动作转换点；优先保留 verb/noun 发生变化的点。
- Assembly101：相邻 fine/coarse action 的转换点；同一 take 只选一个预设主视角。
- Perception Test/Charades：动作段起止点，并结合 parent object/object track 判断候选 Subject。
- Ego-Exo4D/COIN/ActivityNet/YouCook2：使用相邻 keystep 或 temporal segment 的起止点，闭源 VLM 只负责判断该点是否是有效 GEB+ 状态边界并局部细化。

候选点处理规则：

1. 合并相距小于 0.3--0.5 秒且语义相同的候选。
2. 过滤纯镜头切换、淡入淡出、字幕出现和摄像机抖动，除非它们确实造成 GEB+ 定义中的 Subject/Color 变化。
3. 候选点距视频首尾至少保留足够的 before/after 可见上下文。
4. 一个时间戳默认只保留一个 dominant subject；只有两个主体存在独立且明确的状态变化时才产生 A/B 两条样本。

### 4.4 API 输入包

每个候选不直接上传整段长视频，而构造：

- 12--20 秒全局 clip，用于理解动作上下文。
- 以候选点为中心的局部高帧率帧序列，例如 `t±2s`、4--8 fps。
- `t-1.0, t-0.5, t-0.2, t+0.2, t+0.5, t+1.0` 等高分辨率锚点帧。
- 候选来源、原始时间边界和可选 subject/object hints。

原始 action caption 只作为候选生成 hint，不应直接拼入最终 caption prompt，否则模型可能复制文本而不是观察视频。

原生视频 API 可能默认约 1 fps 取样，足以理解全局语义但不足以可靠识别快速或亚秒状态变化，因此时间戳优先来自原始标注/本地 proposer，闭源模型只在有限窗口内细化；必要时使用高帧率接触表或减速局部 clip。

### 4.5 VLM 标注调用

按视频或 30--60 秒 chunk 批处理多个候选，复用同一上传文件，避免每条样本重复编码和上传。一次请求最多放入少量候选，防止主体串扰。

建议强制结构化输出：

```json
{
  "candidate_id": "...",
  "valid_boundary": true,
  "refined_timestamp": 7.54,
  "label": "Change of Action",
  "subject": "man in black shirt",
  "status_before": "banging his head",
  "status_after": "standing straight and raising his hand",
  "evidence_before_time": 7.1,
  "evidence_after_time": 7.9,
  "uncertainty": []
}
```

Prompt 的核心约束：

1. Boundary 是同一 dominant subject 两个相邻状态的分割点。
2. Subject 必须是视觉上可区分的名词短语，不使用仅靠视频外知识得到的身份。
3. Before/After 只描述相邻区间中可见状态，不写原因、推测、意图或未来。
4. 每个 status 必须能由给定证据帧单独支持。
5. 主体出现/消失时严格使用 GEB+ 的 `/0`、`/1` 约定。
6. `label` 只能从五类中选择：`Change of Action`、`Change of Subject`、`Change of Object`、`Change of Color`、`Multiple Changes`。
7. 看不清、无真实变化、只有镜头切换或 Subject 不可区分时输出 `valid_boundary=false`，不能勉强生成。

温度建议保持低值，保存 provider、model snapshot、prompt version、schema version、request ID、输入哈希和原始响应。不要把模型自己报告的 confidence 当成质量真值。

### 4.6 独立验证

验证器不读取生成器的自由推理，只接收视觉输入和候选标注，分别判断：

```text
boundary_visible
timestamp_aligned
subject_distinguishable
subject_identity_consistent
status_before_entailed
status_after_entailed
change_type_correct
camera_cut_only
```

关键样本采用另一模型系列或另一种视觉输入表示进行交叉验证，降低同模型自洽但错误的问题。验证器必须返回离散 verdict 和证据时间，不只返回一个未校准的总分。

建议三级路由：

- 高置信：规则全部通过，生成器和验证器一致，自动接收。
- 不确定：时间偏差、Subject 指代或 status entailment 有一项不稳定，进入第二次模型审核或人工审核。
- 拒绝：无真实变化、只有镜头切换、主体不可辨、前后状态不可见、JSON 无法修复。

### 4.7 确定性规则质检

- `0 <= prev_timestamp < timestamp < next_timestamp <= duration`。
- 相邻 accepted boundary 排序后再计算 `prev_timestamp` 和 `next_timestamp`；第一项为 clip 起点，最后一项为 clip 终点。
- 同一视频内 `boundary_id` 唯一；同一时间戳的 A/B 标注共享 prev/next。
- Subject 非空且是名词短语；两个 status 非空且不能仅为同义复述。
- `/0`、`/1` 只能用于主体出现/消失。
- 过滤 API 拒答、免责声明、Markdown、额外字段和格式污染。
- 使用视频指纹、时间邻近、Subject 归一化和文本 embedding 四级去重。
- 限制单一来源、动作类别、participant、take 和 subject 模板的最大占比。

### 4.8 人工审计与质量门槛

先做 1K--2K 的平衡 pilot，覆盖所有来源、五类边界、短/长动作、第一/第三视角和不同质量档。人工逐项判断 Subject、Before、After 和 timestamp，而不是只判断句子是否通顺。

正式生成后：

- 对全部低置信和分歧样本人工审核。
- 对自动接收样本做约 2% 的按来源/类型/模型版本分层随机抽查。
- 设定 go/no-go 目标，例如完整三元组正确率和时间对齐率达到预先约定阈值；阈值未达到则调整 prompt/路由并重新标注，不通过增加数量掩盖质量问题。
- 保存每轮人工审计的错误 taxonomy，用于比较 prompt/model 版本。

## 5. GEB+ 输出与 provenance

正式训练 JSON 保持 GEB+ 字段不变：

```json
{
  "video_id": [
    {
      "boundary_id": "video_id_0A",
      "timestamp": 7.541,
      "prev_timestamp": 0.0,
      "next_timestamp": 9.95,
      "label": "Change of Action",
      "subject": "man in black shirt",
      "status_before": "banging his head",
      "status_after": "standing straight and raising his hand",
      "action_of_cause": null,
      "caption": "Subject: man in black shirt //Status_Before: banging his head //Status_After: standing straight and raising his hand"
    }
  ]
}
```

`action_of_cause` 在没有可靠来源标注时保持 `null`。`caption` 由三个结构化字段确定性拼接，不让模型单独生成。

另外输出不参与训练的 `provenance.jsonl`：

```text
boundary_id, source_dataset, source_video_id, source_split,
license_id, source_annotation_id, video_sha256, clip_start/end,
candidate_source, raw/refined_timestamp, generator_model/version,
validator_model/version, prompt_hash, request_id, rule_results,
human_review, generation_time
```

## 6. 数据划分与泄漏控制

- 新合成数据只并入训练集，不修改 GEB+ 官方 validation/test。
- 按 source video、participant、take 和近重复 cluster 分组后划分，不能按 boundary 随机拆分。
- 不使用各候选数据集的 test split 生成训练标注。
- 与 GEB+ 的 Kinetics 视频按 YouTube ID、文件哈希和视觉指纹去重。
- Assembly101 多视角视频按 take 分组；同一动作的不同相机不能跨 split，也不应全部计入训练规模。

## 7. 成本与吞吐设计

不预先写死金额，使用实际 provider 价格和 pilot token 日志计算：

```text
总成本 = 上传/存储成本
       + 生成请求输入 token × 输入单价
       + 生成输出 token × 输出单价
       + 验证请求成本
       + 重试与人工审核成本
```

成本优化顺序：

1. 原标注/本地模型生成候选，避免 API 无约束扫长视频。
2. 一个视频请求同时处理多个候选并复用上传文件。
3. 生成使用批处理 API；失败按 request ID 幂等重试。
4. 全局理解用低媒体分辨率，局部证据帧用高分辨率。
5. 只对不确定样本执行第二次生成或强模型验证。
6. 每完成 5K--10K 样本冻结一个 shard，统计接受率、错误类型和真实单位成本后再扩容。

每个请求使用确定性的 `request_key = hash(video_sha256, candidate_time, prompt_version, model_version)`，保证断点续跑不会重复计费或产生重复样本。

## 8. 实施阶段

### Phase 0：定义冻结

- 重算本地 GEB+ 唯一样本数和五类分布。
- 冻结 JSON schema、唯一键、时间精度、Subject 和 `/0`/`/1` 规则。
- 完成数据许可和 API 数据保留审查。

### Phase 1：2K pilot

- Perception Test、EPIC-KITCHENS、Assembly101、Charades 各约 500 个候选。
- 比较 2--3 种输入包和 prompt，而不是先比较大量模型。
- 人工全审，确定 timestamp 容差、接受门槛和单位成本。

### Phase 2：20K pilot

- 测试批处理、缓存、重试、provenance、去重和分层审计。
- 检查来源/动作/主体/边界类型分布，控制厨房和装配偏置。
- 用小规模训练实验验证标签是否可学习，实验由用户在服务器执行。

### Phase 3：扩到 220K--260K

- 每 10K shard 进行自动统计和人工抽查。
- prompt/model 变化必须创建新版本，不能静默覆盖旧响应。
- 质量下降时停止扩容并回滚到上一稳定版本。

### Phase 4：可选扩到 400K

- 只加入通过许可审查的新来源或自采/明确 CC 授权视频。
- 优先补齐欠缺的 Change of Subject/Object/Color 和非厨房、非装配场景，而不是继续堆积 Change of Action。

## 9. 主要风险

1. **许可风险**：把受限视频上传闭源 API 本身可能构成向第三方提供数据；必须先解决。
2. **时间精度风险**：原生视频 API 的低帧率采样不能替代 frame-level boundary proposer。
3. **确认偏差**：把原 action label 放入 caption prompt 会让模型复述 hint，而非视觉验证。
4. **同模型偏差**：同一个 VLM 生成并验证会高估正确率。
5. **规模虚高**：多人复述、多视角、同义改写和近邻时间戳不能当作独立样本。
6. **领域偏置**：EPIC/Assembly/Ego4D 会让训练集过度集中于手物交互和第一视角。
7. **评测泄漏**：Kinetics、公开 benchmark test split 或跨视角同 take 泄漏会夸大效果。

## 10. 主要资料

- [GEB+ ECCV 2022 论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950703.pdf)
- [Something-Something V2 官方页面](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
- [Something-Something Research License](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/jester_something_something_exercise_research_license_final_qti_28jul2022.pdf)
- [Ego4D Hands & Objects](https://ego4d-data.org/docs/benchmarks/hands-and-objects/)
- [Ego4D 数据访问说明](https://ego4d-data.org/docs/start-here/)
- [EPIC-KITCHENS-100](https://epic-kitchens.github.io/2020-100)
- [EPIC-KITCHENS-100 annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
- [Assembly101 论文](https://arxiv.org/abs/2203.14712)
- [Perception Test 官方仓库](https://github.com/google-deepmind/perception_test)
- [Charades 官方页面](https://prior.allenai.org/projects/charades)
- [COIN 论文](https://arxiv.org/abs/1903.02874)
- [VidSitu 官方页面](https://vidsitu.org/)
- [Gemini API Video Understanding](https://ai.google.dev/gemini-api/docs/video-understanding)
- [Gemini Batch API](https://ai.google.dev/gemini-api/docs/batch-api)
- [Vertex AI 数据保留说明](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/vertex-ai-zero-data-retention)
