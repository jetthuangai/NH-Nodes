<p align="center"><img src="../assets/banner.jpg" alt="NH-Nodes cho ComfyUI" width="100%"></p>

<p align="center"><a href="../en/README.md">English</a> · <b>Tiếng Việt</b> · <a href="../../README.md">Trang dự án</a></p>

# Hướng dẫn NH-Nodes

NH-Nodes bổ sung **75 node** cho ComfyUI, phục vụ công việc ảnh và sản xuất: nạp và lưu bộ ảnh, chia ô (tile) và ghép ảnh,
quản lý độ phân giải cho các model hiện nay, công cụ mask, phân đoạn trang phục và khuôn mặt, mask của bất cứ thứ gì bạn gọi tên
bằng SAM 3 qua một công thức ngắn, tạo mask VTON, rẽ nhánh logic, xử lý prompt và văn bản, tiện ích batch và bộ nạp model theo
chỉ số.

Tài liệu này giải thích từng node: dùng để làm gì, từng đầu vào và đầu ra, và cách node xử lý ở những chi tiết quan trọng.
Phần cài đặt và hướng dẫn nhanh nằm ở [trang dự án](../../README.md#installation).

> **Yêu cầu cho SAM 3:** SAM3 Segment cần GPU NVIDIA có CUDA, gói `sam3` của Meta và checkpoint `sam3.pt`, những thứ bạn
> tự cài và tự tải. Mọi node khác chạy được mà không cần chúng. Cách thiết lập:
> [Trước khi bắt đầu với SAM 3](vision.md#trước-khi-bắt-đầu-với-sam-3).

## Cách đọc thẻ node

Mỗi phần node bắt đầu bằng một thẻ được vẽ trực tiếp từ định nghĩa thật của node:

<p align="center"><img src="../assets/nodes/NH_ImageTile.svg" alt="Thẻ node mẫu" width="360"></p>

- **Thanh tiêu đề** — tên hiển thị, bên dưới là mã node (gõ mã này vào ô tìm node) và nhóm menu trong `NH-Nodes`.
- **Chấm bên trái** là đầu vào để nối dây, **chấm bên phải** là đầu ra. Màu chấm cho biết kiểu dữ liệu: IMAGE xanh dương,
  MASK xanh lá, LATENT hồng, MODEL tím, CLIP vàng, VAE đỏ, số (INT, FLOAT) xanh dương nhạt, chữ (STRING) xanh lá nhạt,
  BOOLEAN hồng phấn, kiểu bất kỳ (`*`) màu xám, và dữ liệu riêng của NH (thông tin tile, thông tin resize, instance…) màu hổ phách.
- **Các hàng bo tròn** là widget kèm giá trị mặc định. `opt` đánh dấu đầu vào tùy chọn; `≡` sau một đầu ra nghĩa là đầu ra
  đó trả về danh sách.
- Chấm **xanh lá** trên tiêu đề là node thường; chấm **cam** là node đầu ra (output node), luôn chạy kể cả khi không có gì
  nối vào nó.

## Các nhóm node

| Nhóm | Trang | Node |
|---|---|---|
| Ảnh — nạp & lưu | [image-io](image-io.md) | Load Image Info · Load Images Folder · Load images matching · Save Image Path · NH Save Large Image |
| Ảnh — ghép & chia ô | [image-compose](image-compose.md) | Image Tile · Image Untile · Image Grid Composite · Layer Layout Composite · Layer Stack Composite · Image Compare · Image Label · Image Resize Unit |
| Ảnh — xem ảnh lớn | [image-preview](image-preview.md) | NH Large Image Preview · NH Large Image Compare · NH Large Preview Cache Manager |
| Ảnh — chỉnh sửa theo mask | [image-retouch](image-retouch.md) | NH Mask-Aware Resize · NH Agnostic Image Generator · NH Simple Face Paste |
| Độ phân giải | [resolution](resolution.md) | NH Smart Resolution Picker · NH Smart Ratio Image Resize · NH Ratio Preset Image Resize · NH Remove Resize Padding · NH Restore Image Geometry |
| Mask | [mask](mask.md) | Mask Morphology · Mask Properties · NH Create Box Mask · NH Mask Aspect Ratio Match |
| Vision | [vision](vision.md) | Garment Segment · Portrait Segment · SAM3 Segment · Segs Select · Segs Compose · Segs To Masks · Segs To Impact SEGS · Segs Preview |
| VTON | [vton](vton.md) | VTON Ultimate Processor |
| Logic | [logic](logic.md) | Compare · Logic Gate · If/Else · Switch N · Any Switch · Any Branch Switch · Gate Switch · Value Match Index · Math Eval · Random Choice |
| Văn bản | [text](text.md) | Prompt Template · Prompt Scheduler · String Operations · Prompt Join · Text Split · Regex Extract · Text Index · Text Concatenate · Text Split Lines · Text Random Line |
| Batch & danh sách | [batch-lists](batch-lists.md) | Batch Index · Batch Merge · Counter · List Create · List Index · List Filter · Any List Split |
| Bộ nạp model | [loaders](loaders.md) | Load LoRA Model Index · Load LoRA Clip Index · Load Diffusion Model Index · Load VAE Index · Load Clip Index |
| Tiện ích | [utils](utils.md) | Multi-Slider (FLOAT) · Multi-Slider (INT) · Boolean Switch · Pack Universal · Unpack Universal · Universal Slider Builder *(đã ngừng dùng)* |

## Các khái niệm dùng xuyên suốt

- **Ảnh và mask.** Một `IMAGE` là một batch (lô ảnh) với giá trị từ 0 đến 1 (RGB, đôi khi RGBA); một `MASK` là batch các
  bản đồ một kênh, 1 là bên trong, 0 là bên ngoài, giá trị ở giữa là vùng chuyển mềm. Phần lớn node NH xử lý mọi ảnh trong
  batch; đầu vào chỉ có một ảnh sẽ được dùng lại cho từng ảnh của batch lớn hơn.
- **Nhánh lười (lazy).** Các node rẽ nhánh (If/Else, Switch N, Any Switch, Random Choice) chỉ yêu cầu ComfyUI tính nhánh
  được chọn, nên các nhánh còn lại — kể cả một sampler thứ hai — không bao giờ chạy. Các node kiểu cổng (gate) thì *chặn*
  một nhánh: các node phía sau đầu ra bị chặn được bỏ qua, không gây lỗi.
- **Seed dùng đúng giá trị.** Mọi node có seed cho cùng kết quả với cùng seed, kể cả seed 0. Dùng *control after generate*
  của ComfyUI nếu bạn muốn seed thay đổi sau mỗi lần chạy.
- **Đường dẫn.** Các ô thư mục nhận đường dẫn tuyệt đối hoặc đường dẫn tương đối so với thư mục `input`/`output` của
  ComfyUI; chi tiết ở [image-io](image-io.md#đường-dẫn-và-loại-tệp).
- **Resize data.** Các node độ phân giải trả về `resize_data` ghi lại ảnh đã được cắt hay thêm viền thế nào, để đưa kết quả
  về đúng khung ảnh gốc bằng [NH Restore Image Geometry](resolution.md#nh-restore-image-geometry).
- **Instance.** SAM3 Segment trả về mỗi thứ nó tìm thấy như một instance riêng, có nhãn, điểm và mask, trong một giá trị
  `NH_SEGS`. Các node Segs chọn instance, ghép chúng bằng công thức và đổi chúng thành mask; xem
  [Các instance: kiểu NH_SEGS](vision.md#các-instance-kiểu-nh_segs).
- **Giấy phép.** Mọi node theo giấy phép MIT, trừ các phần ghi trong [THIRD_PARTY_NOTICES](../../THIRD_PARTY_NOTICES.md):
  VTON Ultimate Processor và Portrait Segment chỉ dùng cho mục đích **phi thương mại**, còn model của Garment Segment
  không công bố giấy phép. SAM3 Segment dùng SAM 3 của Meta theo SAM License, giấy phép không có điều khoản cấm thương mại
  nhưng cấm các mục đích quân sự, hạt nhân, gián điệp, súng và vũ khí bất hợp pháp; bạn tự cài gói và tự tải checkpoint.

## Xem thêm

- [Changelog](../CHANGELOG.md) — thay đổi của từng phiên bản, gồm cả các thay đổi về hành vi.
- [Sơ đồ cổng logic](../logic_gate_diagram.html) — bảng chân trị trực quan cho node Logic Gate (tải về và mở bằng trình
  duyệt).
- [Third-party notices](../../THIRD_PARTY_NOTICES.md) — giấy phép của mã nguồn đi kèm, các model được tải về và SAM 3.
