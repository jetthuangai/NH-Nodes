[← Về trang hướng dẫn](./README.md)

# Loaders (trình tải model)

Năm node loader, mỗi node chứa tối đa 64 file model và chỉ tải đúng một file trong số đó, được chọn bằng số thứ tự hoặc bằng công tắc bật/tắt. Dùng chúng để đổi LoRA, diffusion model, VAE hoặc text encoder ngay trong một node, bằng tay hoặc tự động với `Counter (NH)`. Bạn tìm chúng trong menu node tại **NH-Nodes → Loaders**.

Hãy đọc [Cách hoạt động chung của các loader](#cách-hoạt-động-chung-của-các-loader) trước. Mục đó nói về những gì cả năm node dùng chung: slot, cách chọn bằng `index` và `boolean`, chuyện gì xảy ra khi chưa chọn gì, và những gì được giữ trong bộ nhớ.

**Nội dung trang**

| Node | Dùng để |
|---|---|
| [Load LoRA Model Index (NH)](#load-lora-model-index-nh) | Áp một LoRA, chọn từ danh sách tối đa 64, lên `MODEL` |
| [Load LoRA Clip Index (NH)](#load-lora-clip-index-nh) | Áp một LoRA, chọn từ danh sách, lên `MODEL` và `CLIP` |
| [Load Diffusion Model Index (NH)](#load-diffusion-model-index-nh) | Tải một diffusion model từ danh sách |
| [Load VAE Index (NH)](#load-vae-index-nh) | Tải một VAE từ danh sách |
| [Load Clip Index (NH)](#load-clip-index-nh) | Tải một file CLIP hoặc text encoder từ danh sách |

---

## Cách hoạt động chung của các loader

Cả năm node dùng chung các điều khiển và quy tắc. Bạn chỉ cần đọc mục này một lần; các mục node bên dưới chỉ nêu những điểm khác biệt.

### Slot

Mỗi loader có tối đa 64 slot. Một slot gồm một danh sách file và một công tắc bật/tắt (`enable_N`). Widget số lượng quyết định có bao nhiêu slot được hiển thị và sử dụng, mặc định là 5. Các slot vượt quá số lượng này bị ẩn và bị bỏ qua, nhưng vẫn giữ file đã chọn, nên tăng số lượng lên là chúng hiện lại.

| Node | Widget số lượng | Danh sách file | File lấy từ |
|---|---|---|---|
| `Load LoRA Model Index (NH)` | `lora_count` | `lora_1` … `lora_64` | `models/loras` |
| `Load LoRA Clip Index (NH)` | `lora_count` | `lora_1` … `lora_64` | `models/loras` |
| `Load Diffusion Model Index (NH)` | `model_count` | `diffusion_model_1` … `diffusion_model_64` | `models/diffusion_models` và `models/unet` |
| `Load VAE Index (NH)` | `vae_count` | `vae_1` … `vae_64` | `models/vae` |
| `Load Clip Index (NH)` | `clip_count` | `clip_1` … `clip_64` | `models/text_encoders` và `models/clip` |

Các danh sách này cũng gồm cả những thư mục thêm trong `extra_model_paths.yaml`. Để thấy file bạn thêm lúc ComfyUI đang chạy, hãy làm mới định nghĩa node (nhấn phím `R`). Nút `update …` ở cuối mỗi node chỉ vẽ lại việc slot nào đang hiển thị; nó không quét lại thư mục.

### Chọn slot: `select_mode`

| | `index` (mặc định) | `boolean` |
|---|---|---|
| Bạn chọn bằng | số trong `index`, trong đó 1 là slot đầu tiên | các công tắc `enable_N` |
| Hiển thị trên node | `index` | công tắc của các slot đang hiển thị (`index` bị ẩn) |
| Phù hợp để | tự động hóa: nối `Counter (NH)`, `Value Match Index (NH)` hoặc `Multi-Slider (INT)` vào `index` | đổi nhanh bằng tay: bật slot bạn muốn |
| Nhiều công tắc cùng bật | — | dùng công tắc có số nhỏ nhất trong phạm vi số lượng (giao diện của node chỉ cho bật một công tắc tại một thời điểm) |

### Khi chưa chọn được slot nào

Coi như chưa chọn gì khi:

- ở chế độ `index`, `index` nhỏ hơn 1 hoặc lớn hơn số lượng;
- ở chế độ `boolean`, không có công tắc nào trong phạm vi số lượng được bật;
- slot được chọn chưa có file.

Khi đó node không tải gì và:

- đầu ra `model`, `clip` hoặc `vae` của nó bị **chặn** (ComfyUI gọi đây là `ExecutionBlocker`). Mọi node cần đầu ra đó đều bị bỏ qua âm thầm, không có thông báo lỗi, nên lần chạy có thể kết thúc mà không tạo ra ảnh nào;
- `filename` rỗng và `index` là `-1`. Hai đầu ra này không bị chặn.

> **Lưu ý:** Node mới có các slot trống. Hãy chọn file ở slot bạn dùng, nếu không các node phía sau loader sẽ không chạy.

### Những gì được giữ trong bộ nhớ

ComfyUI chạy lại một node mỗi khi một đầu vào của nó thay đổi. Một loader có tới 64 danh sách file và 64 công tắc, nên sửa bất kỳ slot nào, kể cả slot bạn không dùng, cũng được tính là một thay đổi. Để tránh phải tải lại một file lớn mỗi lần như vậy, các loader dùng lại thứ chúng đã tải:

- **`Load Diffusion Model Index (NH)`, `Load VAE Index (NH)` và `Load Clip Index (NH)`** trả lại model mà ComfyUI vẫn đang giữ. Sửa một slot không được chọn sẽ trả về model đang có trong bộ nhớ và không đọc file nào. Model được tải lại khi bạn chọn file khác, đổi `weight_dtype` (diffusion model) hoặc `type` (CLIP), hoặc khi file trên ổ đĩa bị thay thế.
- **Node không tự giữ model.** Node chỉ giữ một tham chiếu yếu (weak reference), nên cache của chính ComfyUI quyết định model tồn tại bao lâu. Khi cache theo áp lực RAM (RAM-pressure) mặc định của ComfyUI giải phóng model vì bộ nhớ sắp hết, lần chạy kế tiếp sẽ tải lại model từ ổ đĩa. Với `--cache-classic` (được bật khi dùng `--high-ram`), node giữ một tham chiếu thông thường, vì cache đó bỏ kết quả ngay khi workflow không còn cần đến.
- **Mỗi node chỉ giữ một model.** Tải file khác thì model trước được giải phóng trước, nên hai model lớn không bao giờ nằm cùng lúc trong bộ nhớ. Vì vậy chuyển A, B, A sẽ phải tải từ ổ đĩa mỗi lần. Xóa execution cache của ComfyUI cũng giải phóng model đang được giữ: hãy dùng **Unload Models and Execution Cache** (trong ComfyUI-Manager lệnh này tên là **Free model and node cache**).
- **Các node LoRA** giữ file LoRA cuối cùng trong bộ nhớ và áp nó lên `model` đầu vào ở mỗi lần chạy. File chỉ được đọc lại khi bạn chọn một LoRA khác.

### Điều khiển `index` tự động

| Nguồn | Nó cho bạn |
|---|---|
| `Counter (NH)` | Mỗi lần queue một slot. Hãy đặt `start` của nó là `1`. Xem [Batch & Lists](./batch-lists.md). |
| `Value Match Index (NH)` | Vị trí (tính từ 1) của giá trị khớp với đầu vào của bạn, hoặc `-1` khi không có giá trị nào khớp (nghĩa là không chọn gì). Xem [Logic](./logic.md). |
| `Multi-Slider (INT)` | Số slot bạn đặt bằng tay trên một bảng điều khiển gọn. Xem [Utils](./utils.md). |

---

## Load LoRA Model Index (NH)

<p align="center"><img src="../assets/nodes/NH_LoraModelIndex.svg" alt="Load LoRA Model Index (NH) node" width="360"></p>

Áp một LoRA, chọn từ tối đa 64 slot, lên `MODEL`.

**Dùng khi** bạn muốn so sánh nhiều LoRA trên cùng một prompt mà không phải nối lại dây, hoặc muốn giữ các LoRA yêu thích trong cùng một node.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `select_mode` | `COMBO` | `index` | Cách chọn slot: `index` hoặc `boolean`. |
| `model` | `MODEL` | — | Model cần áp LoRA. |
| `index` | `INT` | `1` | Slot được dùng ở chế độ `index`. 1 là slot đầu tiên. Phạm vi từ 1 đến 64. |
| `lora_count` | `INT` | `5` | Số slot được hiển thị và sử dụng. Phạm vi từ 1 đến 64. |
| `strength_model` | `FLOAT` | `1.0` | Mức LoRA tác động lên model. Phạm vi từ -100 đến 100, bước 0.01. Giá trị âm đảo ngược tác dụng của LoRA. `0` bỏ qua LoRA. |
| `lora_1` … `lora_64` | `COMBO` | trống | Tùy chọn. File LoRA của từng slot, ví dụ `my_style.safetensors`. |
| `enable_1` … `enable_64` | `BOOLEAN` | `false` | Tùy chọn. Công tắc bật/tắt của từng slot, dùng ở chế độ `boolean`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `model` | `MODEL` | Model đã áp LoRA. Bị chặn khi chưa chọn gì. |
| `filename` | `STRING` | Tên file LoRA như hiển thị trong danh sách slot. Rỗng khi chưa chọn gì hoặc khi `strength_model` là `0`. |
| `index` | `INT` | Slot đã dùng, hoặc `-1` khi chưa chọn gì. |

**Cách hoạt động**

- Khi `strength_model` bằng `0`, LoRA không được tải. `model` đi qua nguyên vẹn và `filename` rỗng. Đây là cách nhanh để tắt một LoRA mà vẫn giữ slot của nó.
- Node áp LoRA lên model đi vào, nên hãy đặt nó sau `Load Checkpoint` và trước sampler.
- Cách chọn slot, trường hợp chưa chọn gì và bộ nhớ hoạt động như mô tả trong [Cách hoạt động chung của các loader](#cách-hoạt-động-chung-của-các-loader).

**Ví dụ — thử các LoRA, mỗi lần queue một LoRA, và gắn nhãn cho từng kết quả**

```text
Counter (NH) → Load LoRA Model Index (NH) → KSampler → VAE Decode → Image Label (NH) → Save Image
```

- `Counter (NH)`: `start` 1, `step` 1, `max_value` bằng số LoRA. Nối `current` vào `index`.
- `Load LoRA Model Index (NH)`: đặt `lora_count` và chọn một LoRA cho mỗi slot. Nối `filename` vào `label_text` của `Image Label (NH)` (xem [Image Compose](./image-compose.md)) để in tên LoRA lên từng ảnh.
- Đặt batch count bằng số LoRA rồi nhấn Run.

---

## Load LoRA Clip Index (NH)

<p align="center"><img src="../assets/nodes/NH_LoraClipIndex.svg" alt="Load LoRA Clip Index (NH) node" width="360"></p>

Áp một LoRA, chọn từ tối đa 64 slot, lên cả `MODEL` và `CLIP`.

**Dùng khi** LoRA cũng cần thay đổi cách prompt được đọc, tức là `CLIP` cần được áp LoRA cùng với `MODEL`.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `select_mode` | `COMBO` | `index` | Cách chọn slot: `index` hoặc `boolean`. |
| `model` | `MODEL` | — | Model cần áp LoRA. |
| `clip` | `CLIP` | — | Text encoder cần áp LoRA. |
| `index` | `INT` | `1` | Slot được dùng ở chế độ `index`. 1 là slot đầu tiên. Phạm vi từ 1 đến 64. |
| `lora_count` | `INT` | `5` | Số slot được hiển thị và sử dụng. Phạm vi từ 1 đến 64. |
| `strength_model` | `FLOAT` | `1.0` | Mức LoRA tác động lên model. Phạm vi từ -100 đến 100, bước 0.01. |
| `strength_clip` | `FLOAT` | `1.0` | Mức LoRA tác động lên CLIP. Phạm vi từ -100 đến 100, bước 0.01. |
| `lora_1` … `lora_64` | `COMBO` | trống | Tùy chọn. File LoRA của từng slot. |
| `enable_1` … `enable_64` | `BOOLEAN` | `false` | Tùy chọn. Công tắc bật/tắt của từng slot, dùng ở chế độ `boolean`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `model` | `MODEL` | Model đã áp LoRA. Bị chặn khi chưa chọn gì. |
| `clip` | `CLIP` | CLIP đã áp LoRA. Bị chặn khi chưa chọn gì. |
| `filename` | `STRING` | Tên file LoRA như hiển thị trong danh sách slot. Rỗng khi chưa chọn gì hoặc khi cả hai mức đều là `0`. |
| `index` | `INT` | Slot đã dùng, hoặc `-1` khi chưa chọn gì. |

**Cách hoạt động**

- Khi cả hai mức đều bằng `0`, LoRA không được tải. `model` và `clip` đi qua nguyên vẹn và `filename` rỗng.
- Hãy dùng `Load LoRA Model Index (NH)` nếu bạn chỉ muốn áp LoRA lên model.

**Ví dụ — đổi LoRA bằng tay với công tắc**

```text
Load Checkpoint → Load LoRA Clip Index (NH) → KSampler / CLIP Text Encode
```

Đặt `select_mode` là `boolean` và chọn các LoRA trong các slot. Bật công tắc của slot bạn muốn. Bật một công tắc sẽ tắt các công tắc còn lại. Để tất cả công tắc tắt thì mọi thứ phía sau loader bị chặn.

---

## Load Diffusion Model Index (NH)

<p align="center"><img src="../assets/nodes/NH_DiffusionModelIndex.svg" alt="Load Diffusion Model Index (NH) node" width="360"></p>

Tải một diffusion model (model chính tạo ảnh hoặc video), chọn từ tối đa 64 slot.

**Dùng khi** bạn muốn so sánh nhiều base model hoặc bản fine-tune trong cùng một workflow, hoặc đổi giữa các độ chính xác của cùng một model, mà không phải nối lại dây.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `select_mode` | `COMBO` | `index` | Cách chọn slot: `index` hoặc `boolean`. |
| `index` | `INT` | `1` | Slot được dùng ở chế độ `index`. 1 là slot đầu tiên. Phạm vi từ 1 đến 64. |
| `model_count` | `INT` | `5` | Số slot được hiển thị và sử dụng. Phạm vi từ 1 đến 64. |
| `weight_dtype` | `COMBO` | `default` | Độ chính xác dùng để tải trọng số: `default`, `fp8_e4m3fn`, `fp8_e4m3fn_fast` hoặc `fp8_e5m2`. |
| `diffusion_model_1` … `diffusion_model_64` | `COMBO` | trống | Tùy chọn. File model của từng slot, ví dụ `my_model.safetensors`. |
| `enable_1` … `enable_64` | `BOOLEAN` | `false` | Tùy chọn. Công tắc bật/tắt của từng slot, dùng ở chế độ `boolean`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `model` | `MODEL` | Diffusion model đã tải. Bị chặn khi chưa chọn gì. |
| `filename` | `STRING` | Tên file model như hiển thị trong danh sách slot. Rỗng khi chưa chọn gì. |
| `index` | `INT` | Slot đã dùng, hoặc `-1` khi chưa chọn gì. |

**Cách hoạt động**

- `weight_dtype` có các lựa chọn giống node `Load Diffusion Model` của ComfyUI. `default` tải trọng số theo cách ComfyUI thường làm. Các lựa chọn `fp8_*` tải trọng số ở dạng số thực 8-bit, tốn ít bộ nhớ hơn, và `fp8_e4m3fn_fast` còn bật thêm các tối ưu fp8 của ComfyUI. Widget này được đánh dấu là nâng cao, nên một số phiên bản ComfyUI để nó thu gọn cho đến khi bạn hiện các tùy chọn nâng cao.
- Node chỉ tải diffusion model. Hãy tải text encoder và VAE bằng `Load Clip Index (NH)` và `Load VAE Index (NH)`.
- Model đã tải được dùng lại chừng nào ComfyUI còn giữ nó (xem [Những gì được giữ trong bộ nhớ](#những-gì-được-giữ-trong-bộ-nhớ)). Đổi `weight_dtype` sẽ tải lại model.

**Ví dụ — đổi cả bộ model chỉ bằng một con số**

Đặt các file tương ứng vào cùng số slot của ba loader:

| Slot | Diffusion model | Text encoder | VAE |
|---|---|---|---|
| 1 | `model_a.safetensors` | `encoder_a.safetensors` | `vae_a.safetensors` |
| 2 | `model_b.safetensors` | `encoder_b.safetensors` | `vae_b.safetensors` |

```text
Counter (NH) → Load Diffusion Model Index (NH) → KSampler
Counter (NH) → Load Clip Index (NH) → CLIP Text Encode
Counter (NH) → Load VAE Index (NH) → VAE Decode
```

Nối cùng một `Counter (NH)` (hoặc một `Multi-Slider (INT)`) vào `index` của cả ba. Khi đó một con số đổi cả model, text encoder và VAE cùng lúc.

---

## Load VAE Index (NH)

<p align="center"><img src="../assets/nodes/NH_VAEIndex.svg" alt="Load VAE Index (NH) node" width="360"></p>

Tải một VAE, chọn từ tối đa 64 slot.

**Dùng khi** bạn muốn so sánh cách các VAE khác nhau giải mã cùng một ảnh, hoặc đổi VAE mà không phải nối lại dây.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `select_mode` | `COMBO` | `index` | Cách chọn slot: `index` hoặc `boolean`. |
| `index` | `INT` | `1` | Slot được dùng ở chế độ `index`. 1 là slot đầu tiên. Phạm vi từ 1 đến 64. |
| `vae_count` | `INT` | `5` | Số slot được hiển thị và sử dụng. Phạm vi từ 1 đến 64. |
| `vae_1` … `vae_64` | `COMBO` | trống | Tùy chọn. File VAE của từng slot, ví dụ `my_vae.safetensors`. |
| `enable_1` … `enable_64` | `BOOLEAN` | `false` | Tùy chọn. Công tắc bật/tắt của từng slot, dùng ở chế độ `boolean`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `vae` | `VAE` | VAE đã tải. Bị chặn khi chưa chọn gì. |
| `filename` | `STRING` | Tên file VAE như hiển thị trong danh sách slot. Rỗng khi chưa chọn gì. |
| `index` | `INT` | Slot đã dùng, hoặc `-1` khi chưa chọn gì. |

**Cách hoạt động**

- Giống loader VAE của ComfyUI, node chuyển siêu dữ liệu (metadata) lưu trong file cho VAE và kiểm tra file ngay khi tải. File không chứa VAE dùng được sẽ làm lần chạy dừng tại node này với một lỗi, thay vì thất bại muộn hơn lúc giải mã ảnh. VAE không qua được bước kiểm tra sẽ không được giữ lại.
- VAE đã tải được dùng lại chừng nào ComfyUI còn giữ nó, cho đến khi bạn chọn file khác (xem [Những gì được giữ trong bộ nhớ](#những-gì-được-giữ-trong-bộ-nhớ)).

**Ví dụ — so sánh hai VAE trên cùng một latent**

```text
Load VAE Index (NH) → VAE Decode → Preview Image
```

Nối `vae` vào đầu vào `vae` của `VAE Decode`, và chọn mỗi VAE một slot. Đặt seed của sampler ở chế độ cố định (fixed), chạy một lần với `index` 1 và một lần với `index` 2, rồi so sánh hai ảnh được giải mã.

---

## Load Clip Index (NH)

<p align="center"><img src="../assets/nodes/NH_ClipIndex.svg" alt="Load Clip Index (NH) node" width="360"></p>

Tải một file CLIP hoặc text encoder, chọn từ tối đa 64 slot, và đặt loại CLIP mà file được tải theo.

**Dùng khi** bạn muốn so sánh các text encoder, hoặc đổi giữa các dòng model mà mỗi dòng cần text encoder riêng.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `select_mode` | `COMBO` | `index` | Cách chọn slot: `index` hoặc `boolean`. |
| `type` | `COMBO` | `stable_diffusion` | Loại CLIP dùng để tải file. Danh sách theo phiên bản ComfyUI của bạn, ví dụ `stable_diffusion`, `sd3`, `flux` hoặc `wan`. |
| `index` | `INT` | `1` | Slot được dùng ở chế độ `index`. 1 là slot đầu tiên. Phạm vi từ 1 đến 64. |
| `clip_count` | `INT` | `5` | Số slot được hiển thị và sử dụng. Phạm vi từ 1 đến 64. |
| `clip_1` … `clip_64` | `COMBO` | trống | Tùy chọn. File text encoder của từng slot, ví dụ `my_encoder.safetensors`. |
| `enable_1` … `enable_64` | `BOOLEAN` | `false` | Tùy chọn. Công tắc bật/tắt của từng slot, dùng ở chế độ `boolean`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `clip` | `CLIP` | Text encoder đã tải. Bị chặn khi chưa chọn gì. |
| `filename` | `STRING` | Tên file như hiển thị trong danh sách slot. Rỗng khi chưa chọn gì. |
| `index` | `INT` | Slot đã dùng, hoặc `-1` khi chưa chọn gì. |

**Cách hoạt động**

- Hãy chọn `type` khớp với dòng model mà file thuộc về. Tên loại mà ComfyUI của bạn không nhận ra sẽ quay về `stable_diffusion`.
- Mỗi lần chạy node chỉ tải một file. Với các dòng model cần hai hoặc nhiều file text encoder cùng lúc, hãy dùng các loader dual hoặc triple CLIP có sẵn của ComfyUI.
- CLIP đã tải được dùng lại chừng nào ComfyUI còn giữ nó (xem [Những gì được giữ trong bộ nhớ](#những-gì-được-giữ-trong-bộ-nhớ)). Đổi `type` sẽ tải lại.

**Ví dụ — chọn text encoder bằng tay**

```text
Load Clip Index (NH) → CLIP Text Encode → KSampler
```

Đặt `type` theo dòng model của bạn, chọn một encoder cho mỗi slot, rồi đổi `index` để chuyển.

---

**Xem thêm:** [Batch & Lists](./batch-lists.md) cho `Counter (NH)`, [Utils](./utils.md) cho `Multi-Slider (INT)`, và [Logic](./logic.md) cho `Value Match Index (NH)`.

[← Về trang hướng dẫn](./README.md)
