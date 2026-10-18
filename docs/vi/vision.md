[← Về trang hướng dẫn](./README.md)

# Vision (phân đoạn ảnh)

Tám node cắt từng phần trên ảnh chụp. Hai node dùng các model SegFormer hiểu về con người: trang phục với Garment Segment, và khuôn mặt, tóc, cổ với Portrait Segment. Mỗi node trả về một mask có cùng kích thước với ảnh đầu vào, một ảnh cắt nền có độ trong suốt và một ảnh xem trước mask. Sáu node còn lại dùng SAM 3, model phân đoạn của Meta: SAM3 Segment tìm mọi instance (từng đối tượng riêng lẻ) của bất cứ thứ gì bạn gọi tên bằng tiếng Anh thông thường (người, ly rượu, cửa sổ, bầu trời), và năm node Segs giúp chọn, ghép, xem trước và chuyển đổi những gì nó tìm được. Bạn tìm thấy chúng trong **NH-Nodes → Vision**.

**Trên trang này**

- [Nên dùng node nào?](#nên-dùng-node-nào) — các node SegFormer hay SAM 3
- **Các node SegFormer: trang phục và chân dung**
  - [Trước khi bắt đầu](#trước-khi-bắt-đầu) — model, lần chạy đầu tiên, giấy phép, bộ nhớ
  - [Tinh chỉnh mask](#tinh-chỉnh-mask) — các tùy chọn dùng chung của hai node
  - [Garment Segment (NH)](#garment-segment-nh) — mask trang phục thân trên, thân dưới, giày dép hoặc mũ nón
  - [Portrait Segment (NH)](#portrait-segment-nh) — mask khuôn mặt, tóc, cổ và các bộ phận trên mặt
- **Các node SAM 3: bất cứ thứ gì bạn gọi tên được**
  - [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3) — GPU, gói cài đặt, checkpoint, bộ nhớ, giấy phép
  - [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động) — `balanced` làm được gì trên ảnh 4K đến 8K, và giới hạn ở đâu
  - [Các instance: kiểu NH_SEGS](#các-instance-kiểu-nh_segs) — thứ được chuyển qua lại giữa các node
  - [SAM3 Segment (NH)](#sam3-segment-nh) — tìm mọi instance của các concept bạn gọi tên
  - [Segs Select (NH)](#segs-select-nh) — giữ instance theo nhãn, điểm, kích thước và vị trí
  - [Segs Compose (NH)](#segs-compose-nh) — dựng mask từ nhãn bằng `|`, `-`, `&` và `~`
  - [Segs To Masks (NH)](#segs-to-masks-nh) — mỗi instance một mask, cùng mask hợp và danh sách nhãn
  - [Segs To Impact SEGS (NH)](#segs-to-impact-segs-nh) — đưa các instance cho detailer của Impact Pack
  - [Segs Preview (NH)](#segs-preview-nh) — vẽ các instance lên ảnh của chúng
  - [Workflow với SAM 3](#workflow-với-sam-3) — nền phía sau người, vật lớn nhất, mặt tiền không có cửa sổ
  - [Khi có sự cố với SAM 3](#khi-có-sự-cố-với-sam-3)

## Nên dùng node nào?

| Bạn muốn | Dùng | Cần gì |
|---|---|---|
| Trang phục của một người: áo, quần, giày, mũ | [Garment Segment](#garment-segment-nh) | Một model SegFormer tự tải ở lần chạy đầu. Chạy được trên CPU hoặc GPU. |
| Khuôn mặt, tóc, cổ và các bộ phận trên mặt của một ảnh chân dung | [Portrait Segment](#portrait-segment-nh) | Giống như trên. |
| Bất cứ thứ gì bạn gọi tên được (người, đồ vật, bầu trời, tường, cửa sổ), từng instance riêng, kể cả trên ảnh 4K đến 8K | [SAM3 Segment](#sam3-segment-nh) và các node Segs | GPU NVIDIA, gói `sam3` của Meta và một checkpoint mà bạn tự cài và tự tải. Xem [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3). |

Bạn cũng có thể yêu cầu SAM3 Segment tìm `dress` hay `hair`. Các node SegFormer vẫn là lựa chọn dễ hơn cho các bộ phận trang phục và khuôn mặt cố định của chúng: chúng tự thiết lập và cũng chạy được không cần GPU.

## Trước khi bắt đầu

Phần này nói về Garment Segment và Portrait Segment. SAM3 Segment và các node Segs có phần riêng: [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3).

| | Garment Segment (NH) | Portrait Segment (NH) |
|---|---|---|
| Model | SegFormer huấn luyện cho các lớp trang phục của Fashionpedia | SegFormer tinh chỉnh (fine-tune) trên CelebAMask-HQ cho các bộ phận khuôn mặt |
| Kho Hugging Face | `1038lab/segformer_fashion` | `jonathandinu/face-parsing` |
| Dung lượng tải | khoảng 190 MB | khoảng 340 MB |
| Giấy phép | Người đăng tải không khai báo | Chỉ dành cho nghiên cứu và giáo dục phi thương mại |

**Lần chạy đầu tiên.** Khi chạy lần đầu, mỗi node tải model của nó từ Hugging Face. Mỗi lần tải được cố định (pinned) vào một phiên bản xác định của kho, nên thay đổi về sau ở nguồn không thể làm đổi thứ bạn nhận được. Các tệp được lưu vào `ComfyUI/models/NH-Nodes/`, mỗi model một thư mục con. Nếu đã có sẵn một bản đầy đủ thì node dùng luôn bản đó: trong `models/RMBG/segformer_fashion` với model trang phục, trong `models/face_parsing` hoặc `models/RMBG/segformer_face` với model khuôn mặt, hoặc trong bất kỳ thư mục nào bạn thêm bằng khóa `nh_nodes` trong `extra_model_paths.yaml` (xem ví dụ ở mục [Installation](../../README.md#installation)). Nếu quá trình tải bị gián đoạn, hãy chạy lại node: các tệp đã tải xong được giữ lại.

**Bộ nhớ.** Model được đọc từ đĩa một lần và giữ trong bộ nhớ hệ thống. Nó chỉ chuyển sang GPU trong lúc node chạy, rồi trở về bộ nhớ hệ thống, nên không chiếm VRAM giữa các lần chạy. Trước khi chuyển sang GPU, node nhờ ComfyUI gỡ bớt các model của chính nó nếu VRAM đang eo hẹp.

> **Lưu ý:** Nếu node báo nhãn không như mong đợi (unexpected labels), nghĩa là có một checkpoint khác nằm trong một thư mục mà node tìm. Hãy xóa bản đó, model đúng sẽ được tải về.

> **Giấy phép:** Model đứng sau Portrait Segment chỉ được dùng cho nghiên cứu và giáo dục phi thương mại. Model đứng sau Garment Segment không khai báo giấy phép, nên điều khoản của nó chưa rõ. Mã nguồn của NH-Nodes thuộc giấy phép MIT. Hãy đọc [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) trước khi dùng một trong hai node cho mục đích thương mại.

## Tinh chỉnh mask

Garment Segment và Portrait Segment biến bản đồ nhãn của model thành mask, rồi tinh chỉnh theo thứ tự: `fill_holes`, `mask_expand`, `mask_blur`. Mask luôn có kích thước bằng ảnh đầu vào.

| Tùy chọn | Khoảng | Garment Segment | Portrait Segment | Tác dụng |
|---|---|---|---|---|
| `fill_holes` | bật / tắt | tắt | bật | Lấp các lỗ nằm trọn trong mask. Với trang phục, mặc định là tắt vì nếu bật, cánh tay vắt ngang thân (nằm lọt giữa vùng trang phục) sẽ bị lấp vào mask. |
| `mask_expand` | -64 đến 64 | `0` | `0` | Mở rộng (số dương) hoặc thu hẹp (số âm) mask đúng bấy nhiêu pixel. |
| `mask_blur` | 0 đến 64 | `0` | `0` | Làm mềm (feather) cạnh khoảng bấy nhiêu pixel. |
| `device` | `cuda`, `cpu` | `cuda` | `cuda` | Nơi model chạy. Không có CUDA thì node chuyển sang CPU và in một ghi chú một lần trong console. |

Ba đầu ra dưới đây dùng chung cho cả hai node:

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask với giá trị từ 0 đến 1. Cạnh mềm khi `mask_blur` lớn hơn 0. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. Màu bên ngoài mask được giữ nguyên. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh RGB màu xám, dùng để xem trước hoặc cho các node cần `IMAGE`. |

Batch (lô ảnh) đều dùng được: từng ảnh được phân đoạn lần lượt và các đầu ra giữ nguyên thứ tự batch.

---

## Garment Segment (NH)

<p align="center"><img src="../assets/nodes/NH_GarmentSegment.svg" alt="Garment Segment (NH) node" width="360"></p>

Tạo mask cho một nhóm trang phục, hoặc tất cả các nhóm cùng lúc, trên người trong ảnh.

**Dùng khi** bạn muốn có mask sạch của áo, quần, giày hoặc mũ để inpaint, đổi màu hoặc dùng trong workflow thử đồ.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Ảnh chụp một người. |
| `part` | COMBO | `upper_garment` | `upper_garment`, `lower_garment`, `footwear`, `headwear` hoặc `all_garments`. Xem bảng bên dưới. |
| `attach_parts` | BOOLEAN | `true` | Gộp túi, khóa kéo, nơ, bèo nhún và các chi tiết tương tự chạm vào trang phục đã chọn. |
| `mask_expand` | INT | `0` | Khoảng -64 đến 64. Mở rộng (+) hoặc thu hẹp (-) mask N pixel. |
| `mask_blur` | INT | `0` | Khoảng 0 đến 64. Làm mềm cạnh mask N pixel. |
| `fill_holes` | BOOLEAN | `false` | Lấp các lỗ khép kín. |
| `device` | COMBO | `cuda` | `cuda` hoặc `cpu`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask của phần đã chọn. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh xem trước RGB màu xám. |

**Mỗi `part` gồm những gì**

| `part` | Bao gồm |
|---|---|
| `upper_garment` | Áo sơ mi và áo kiểu, áo thun và áo nỉ, áo len, cardigan, áo khoác, áo gile, áo khoác dài (coat), đầm, jumpsuit, áo choàng (cape), cà vạt, khăn quàng, mũ trùm áo, cổ áo, ve áo, cầu vai, tay áo và đường viền cổ. |
| `lower_garment` | Quần dài, quần short, váy, thắt lưng, khóa thắt lưng, bó chân (leg warmer), quần tất và tất dài. Không gồm giày. |
| `footwear` | Giày và tất. |
| `headwear` | Mũ, băng đô, khăn đội đầu và phụ kiện tóc. |
| `all_garments` | Cả bốn nhóm trong cùng một mask. |

**Cách hoạt động**

- **Đầm và jumpsuit** thuộc `upper_garment`. Không có tùy chọn đầm riêng. Muốn mask áo và quần cùng lúc, hoặc đầm và giày, hãy chọn `all_garments`.
- **Chi tiết đi kèm (attached parts)** gồm túi (pocket), khóa kéo, họa tiết đính (appliqué), hạt cườm, nơ, hoa, tua rua, ruy băng, đinh tán, bèo nhún, kim sa (sequin) và tua (tassel). Khi bật `attach_parts`, một chi tiết chỉ được thêm vào khi nó chạm vào trang phục đã chọn (trong vòng vài pixel). Túi trên áo khoác sẽ nhập vào mask `upper_garment`, còn chiếc nơ trên một chiếc túi xách thì không.
- **Không bao giờ được chọn:** kính, găng tay, đồng hồ, túi xách, ví và ô không thuộc `part` nào.

<p align="center"><img src="../assets/examples/garment-segment.jpg" alt="Mask của Garment Segment phủ lên một ảnh mẫu" width="720"><br><em>Garment Segment trên một ảnh mẫu: các vùng tô màu là mask của những phần đã chọn.</em></p>

**Ví dụ**

```text
Load Image → Garment Segment (NH, upper_garment) → mask → Mask Morphology (NH) → inpaint
```

Hãy mở rộng mask thêm vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh) để trang phục được inpaint có thể phủ lên đường viền cũ.

> **Giấy phép:** model trang phục không khai báo giấy phép. Xem [Trước khi bắt đầu](#trước-khi-bắt-đầu).

---

## Portrait Segment (NH)

<p align="center"><img src="../assets/nodes/NH_PortraitSegment.svg" alt="Portrait Segment (NH) node" width="360"></p>

Tạo mask cho khuôn mặt, tóc, cổ và các bộ phận trên mặt của một ảnh chân dung. Bạn bật những vùng muốn lấy.

**Dùng khi** bạn muốn chỉnh sửa hoặc đổi màu tóc, xử lý da và các bộ phận trên mặt, hoặc bảo vệ khuôn mặt trong khi chỉnh phần còn lại của ảnh.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Một ảnh chân dung. |
| `face` | BOOLEAN | `true` | Vùng khuôn mặt đặc: da, mũi, mắt, lông mày, kính mắt, môi và bên trong miệng. |
| `neck` | BOOLEAN | `true` | Cổ. |
| `hair` | BOOLEAN | `true` | Tóc. |
| `eyes` | BOOLEAN | `true` | Hai mắt. |
| `lips` | BOOLEAN | `true` | Môi trên và môi dưới. |
| `nose` | BOOLEAN | `true` | Mũi. |
| `eyebrows` | BOOLEAN | `true` | Hai lông mày. |
| `ears` | BOOLEAN | `true` | Hai tai. |
| `mouth_interior` | BOOLEAN | `false` | Bên trong miệng, giữa hai môi. |
| `mask_expand` | INT | `0` | Khoảng -64 đến 64. Mở rộng (+) hoặc thu hẹp (-) mask N pixel. |
| `mask_blur` | INT | `0` | Khoảng 0 đến 64. Làm mềm cạnh mask N pixel. |
| `fill_holes` | BOOLEAN | `true` | Lấp các lỗ khép kín. |
| `device` | COMBO | `cuda` | `cuda` hoặc `cpu`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask gộp của mọi vùng đã bật. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh xem trước RGB màu xám. |

**Cách hoạt động**

- Các vùng đã bật được gộp thành một mask. Không bật vùng nào thì mask rỗng.
- `face` đã bao gồm mắt, lông mày, mũi, môi và bên trong miệng. Các tùy chọn đó chỉ làm kết quả thay đổi khi `face` tắt.
- Mũ, hoa tai, dây chuyền và quần áo có trong model nhưng node không cung cấp.

<p align="center"><img src="../assets/examples/portrait-segment.jpg" alt="Mask của Portrait Segment phủ lên một ảnh mẫu" width="720"><br><em>Portrait Segment trên một ảnh mẫu: các vùng tô màu là mask của những vùng đã bật.</em></p>

**Ví dụ: chỉ lấy tóc**

```text
Load Image → Portrait Segment (NH) → mask → đổi màu hoặc inpaint
```

Tắt mọi vùng trừ `hair`. Với các giá trị mặc định, bạn có khuôn mặt, tai, cổ và tóc cùng trong một mask.

> **Mẹo:** Model được tinh chỉnh trên CelebAMask-HQ, một bộ ảnh chụp cận mặt. Nó hoạt động tốt nhất khi khuôn mặt chiếm một phần đáng kể của khung hình. Với khuôn mặt nhỏ trong ảnh lớn, hãy cắt (crop) quanh khuôn mặt trước.

> **Giấy phép:** model face-parsing chỉ dành cho nghiên cứu và giáo dục phi thương mại. Xem [Trước khi bắt đầu](#trước-khi-bắt-đầu).

---

## Trước khi bắt đầu với SAM 3

SAM 3 là model của Meta để tìm vật theo tên: bạn viết thứ mình muốn (`person`, `window`, `wine glass`) và nó trả về mọi instance của thứ đó, mỗi instance có điểm tin cậy (score) và mask riêng. NH-Nodes không kèm model và không bao giờ tự tải nó, nên SAM3 Segment cần bạn chuẩn bị ba thứ, chỉ một lần. Nếu một pack SAM 3 khác đã chạy tốt trong ComfyUI của bạn, bước 2 và 3 thường đã xong: pack đó đã cài gói và đặt `sam3.pt` đúng nơi node này tìm.

**1. GPU NVIDIA có CUDA.** SAM 3 chỉ chạy trên CUDA: mã sam3 chính thức dựng một phần model trên GPU, nên không có chế độ CPU và không có đầu vào `device`. Hãy dự trù khoảng 5 GB VRAM ở mức cao nhất.

**2. Gói `sam3` của Meta.** NH-Nodes không tự cài nó cho bạn. Trong Python đang chạy ComfyUI, hãy chạy hai dòng sau, rồi khởi động lại ComfyUI:

```text
python -m pip install --no-deps git+https://github.com/facebookresearch/sam3@b26a5f330e05d321afb39d01d3d4881f258f65ff
python -m pip install timm ftfy regex iopath
```

Dòng đầu cài đúng commit sam3 mà NH-Nodes đã được thử nghiệm; các commit mới hơn có cùng API ảnh cũng dùng được. Dòng này cần máy bạn có Git. `--no-deps` giữ nguyên numpy của bạn: sam3 yêu cầu numpy dưới 2, điều mà ComfyUI không cần. Trang của Meta cho gói này ghi Python 3.12 trở lên, PyTorch 2.7 trở lên và CUDA 12.6 trở lên. Nếu một pack SAM 3 khác đã cung cấp `sam3`, hãy bỏ qua dòng đầu: cài lại có thể thay thế bản mà pack đó đang dùng.

**3. Checkpoint `sam3.pt`.** Meta chỉ phát trọng số sau khi bạn chấp nhận SAM License của họ, nên NH-Nodes không thể tải giúp bạn. Hãy xin quyền truy cập tại [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), tải `sam3.pt` từ trang đó và đặt vào `ComfyUI/models/sam3/`. Nhấn R trong ComfyUI để làm mới danh sách, rồi chọn tệp trong `checkpoint`. Muốn để ở nơi khác, hãy thêm thư mục đó vào khóa `sam3` trong `extra_model_paths.yaml` (xem tệp này ở mục [Installation](../../README.md#installation)):

```yaml
my_models:
    base_path: D:/ai-models/
    sam3: sam3/                    # checkpoint SAM 3, dùng chung với các pack SAM 3 khác
```

Chỉ checkpoint PyTorch `sam3.pt` dùng được. Bản safetensors trên Hugging Face dành cho thư viện transformers và dùng tên trọng số khác.

**Bộ nhớ.** Trọng số được đọc từ đĩa một lần và giữ trong bộ nhớ hệ thống (khoảng 3.6 GB). Chúng chuyển lên GPU khi SAM3 Segment chạy và trở về sau đó, nên ComfyUI nhận lại VRAM. Trước khi chuyển, node nhờ ComfyUI gỡ bớt các model của chính nó nếu VRAM đang eo hẹp. Hãy bật `keep_model_loaded` để trọng số ở lại trên GPU nếu bạn chạy node liên tục.

> **Giấy phép:** SAM 3 được phát hành theo SAM License của Meta. Giấy phép này miễn phí bản quyền (royalty-free) và không có điều khoản cấm thương mại, nhưng cấm những mục đích thuộc ITAR hoặc bị kiểm soát thương mại (trade controls) cấm, gồm quân sự hay chiến tranh, hạt nhân, gián điệp, súng và vũ khí bất hợp pháp. Mã nguồn của NH-Nodes thuộc giấy phép MIT và không chứa model lẫn gói `sam3`. Hãy đọc [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) trước khi dùng node cho mục đích thương mại.

---

## Cách hai lượt phân tích hoạt động

SAM 3 nhìn một bức ảnh cỡ 1000 px. Trên ảnh 6000 px, đó chỉ bằng một phần sáu bề ngang, nên vật nhỏ và các cạnh mảnh sẽ bị nhòe. Vì vậy SAM3 Segment nhìn hai lượt.

1. **Lượt đầu: cả bức ảnh.** Ảnh được thu nhỏ về đầu vào vuông của model (1008 px) và mã hóa một lần. Mọi concept bạn liệt kê được hỏi trên đúng lần mã hóa đó, rất nhẹ. Kết quả cho mỗi instance một điểm, một khung (box) và một mask đầu tiên, được lấy mẫu lại lên độ phân giải đầy đủ, chỉ trong phạm vi khung của chính nó.
2. **Lượt hai: ảnh cận cảnh của từng instance** (chỉ với `balanced`). Instance được cắt khỏi ảnh độ phân giải đầy đủ kèm một lề (20% cạnh dài của nó, tối thiểu 16 px) và được phân đoạn lại bằng bộ giải mã dùng khung gợi ý (box-prompt decoder) của SAM 3, lấy mask lượt đầu làm gợi ý. Lúc này cạnh của nó đến từ các điểm ảnh gốc.
3. **Cơ chế dự phòng.** Mask cận cảnh được giữ khi nó khớp với instance của lượt đầu trong phạm vi độ nhòe của lượt đầu, nên những thứ mảnh như cột và dây cáp vẫn còn. Khi không khớp, mask lượt đầu được giữ lại, và console cho biết số instance bị như vậy. Các mảnh vụn của mask mới không chạm vào mask lượt đầu bị bỏ đi.

| `detail_level` | Lượt | Dùng khi |
|---|---|---|
| `balanced` (mặc định) | Cả hai | Cạnh sắc nét cho vật nhỏ và vừa, kể cả trên ảnh 4K đến 8K. Khoảng 0.1 giây mỗi instance trên GPU nhanh. |
| `fast` | Chỉ lượt đầu | Ảnh nhỏ, và khi thử nhanh để tìm từ đúng cho concept. |

**`balanced` làm được và không làm được gì.** Vật nhỏ và vừa (một ô cửa sổ, một chiếc ly, một người trong ảnh nhóm) có cạnh sắc nét ngay cả trên ảnh 4K đến 8K, vì mỗi vật được xét trên một ảnh cận cảnh của các điểm ảnh gốc. Giới hạn nằm ở model: SAM 3 mô tả bất cứ thứ gì nó nhìn bằng một lưới khoảng 288 ô. Một đối tượng chiếm gần hết khung hình 8K tự nó là ảnh cận cảnh, nên cạnh của nó chỉ chính xác đến khoảng 1/300 kích thước của nó, tức cỡ 25 px với đối tượng rộng 7000 px. Mức đó ổn cho đường viền của một người hay một tòa nhà, và còn mềm ở tóc, ren và lông. Cạnh ở mức sợi tóc cần một bước tách nền mịn (matting) mà NH-Nodes chưa có; một node Matte Refine đang được dự kiến.

**Đo trên một máy** (NVIDIA RTX 5090): lần chạy đầu nạp model mất khoảng 9 giây. Một ảnh 7360×4912 với các concept `person, wine glass, dress` cho 13 instance trong khoảng 3.6 giây với `balanced`. Một ảnh nội thất 6720×4480 với 8 concept cho 29 instance trong khoảng 6 giây. Mức VRAM cao nhất khoảng 5 GB. GPU khác sẽ cho số khác.

---

## Các instance: kiểu NH_SEGS

`NH_SEGS` là dữ liệu mà SAM3 Segment chuyển cho các node Segs. Với mỗi ảnh trong batch (lô ảnh), nó chứa một danh sách instance. Mỗi instance giữ **nhãn** (concept bạn đã viết), **điểm tin cậy**, một **khung** ôm sát và mask **chỉ trong phạm vi khung đó**. Mask kích thước đầy đủ chỉ được dựng khi một node xuất `MASK`, nên hàng chục instance trên ảnh 8K vẫn nhẹ.

- Các instance xếp theo thứ tự concept bạn viết, và theo điểm trong cùng một concept.
- Các node Segs xử lý từng ảnh trong batch riêng biệt.
- Các node gốc của ComfyUI không đọc được `NH_SEGS`. Để lấy mask ra, hãy dùng các đầu ra `mask`, [Segs Compose](#segs-compose-nh) hoặc [Segs To Masks](#segs-to-masks-nh). Để dùng detailer của Impact Pack, hãy chuyển đổi bằng [Segs To Impact SEGS](#segs-to-impact-segs-nh).

---

## SAM3 Segment (NH)

<p align="center"><img src="../assets/nodes/NH_SAM3Segment.svg" alt="SAM3 Segment (NH) node" width="360"></p>

Tìm mọi instance của những thứ bạn gọi tên và trả về chúng dưới dạng `NH_SEGS`, kèm mask gộp của chúng và số lượng. Muốn xem nó tìm được gì, hãy nối [Segs Preview](#segs-preview-nh).

**Dùng khi** bạn muốn có mask của "mọi người", "mọi cửa sổ", "bầu trời" hay "các ly rượu" trong một bức ảnh, kể cả ảnh 4K đến 8K, mà không phải vẽ gì. Sau đó dùng các [node Segs](#segs-select-nh) để chọn instance, trừ cái này khỏi cái kia, hoặc đổi chúng thành mask thông thường.

Các đầu vào COMBO không nêu mặc định sẽ bắt đầu ở lựa chọn đầu tiên.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Bức ảnh. Node dùng màu (RGB), kênh alpha nếu có sẽ bị bỏ. Mọi ảnh trong batch được xử lý lần lượt. |
| `concepts` | STRING | `person` | Thứ cần tìm: các cụm danh từ tiếng Anh ngắn, mỗi dòng một cụm hoặc cách nhau bằng dấu phẩy (`person, red car, window, sky`). Dòng bắt đầu bằng `#` bị bỏ qua. Mỗi cụm trở thành nhãn của các instance nó tìm được. Hãy viết ít nhất một cụm. |
| `threshold` | FLOAT | `0.5` | Khoảng 0.05 đến 0.95. Độ tin cậy tối thiểu. Thấp hơn thì tìm được nhiều instance hơn và cũng nhiều nhầm lẫn hơn; cao hơn chỉ giữ những cái chắc chắn. |
| `detail_level` | COMBO | `balanced` | `balanced` hoặc `fast`. `balanced` phân đoạn lại mỗi instance lần thứ hai trên ảnh cận cảnh cắt từ ảnh độ phân giải đầy đủ, để vật nhỏ và vừa có cạnh sắc nét. `fast` giữ lượt đầu trên cả bức ảnh. Xem [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động). |
| `max_per_concept` | INT | `0` | Khoảng 0 đến 200. Giữ tối đa bấy nhiêu instance cho mỗi concept, điểm cao nhất trước. `0` giữ tất cả. |
| `checkpoint` | COMBO | — | Một tệp `.pt` hoặc `.pth` của SAM 3 trong `models/sam3` hoặc thư mục khác đã đăng ký với tên `sam3`. Các pack SAM 3 khác dùng chung thư mục này. Khi thư mục chưa có tệp nào, danh sách hiện `sam3.pt` và node cho bạn biết lấy nó ở đâu khi bạn chạy. |
| `keep_model_loaded` | BOOLEAN | `false` | Tắt: sau lần chạy, trọng số trở về bộ nhớ hệ thống và ComfyUI nhận lại VRAM. Bật: chúng ở lại trên GPU (khoảng 3.6 GB), giúp lần chạy sau nhanh hơn, nhưng ComfyUI không dùng được phần VRAM đó trong lúc ấy. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `segs` | NH_SEGS | Các instance của từng ảnh, mỗi instance có nhãn, điểm, khung và mask. Hãy nối nó vào các node Segs. |
| `mask` | MASK | Hợp của mọi instance, một mask cho mỗi ảnh, có kích thước bằng ảnh. |
| `count` | INT | Số instance tìm được, tính trên cả batch. |

<p align="center"><img src="../assets/examples/sam3-segment.jpg" alt="Các instance của SAM3 Segment do Segs Preview vẽ, và ba mask tạo từ chúng trên một ảnh mẫu" width="720"><br><em>SAM3 Segment trên một ảnh mẫu với các concept "person, helmet, face", do Segs Preview vẽ, rồi ba mask tạo từ các instance bằng Segs Compose và Segs Select.</em></p>

**Cách hoạt động**

- **Nhãn.** Concept bạn viết là nhãn của các instance của nó. Các node Segs bỏ qua chữ hoa/thường khi so khớp nhãn, nên cách viết hoa không thành vấn đề.
- **Thứ tự và bản trùng.** Các instance xếp theo thứ tự concept của bạn, điểm cao nhất trước trong mỗi concept. Hai instance của cùng một concept mà mask gần như trùng hoàn toàn (từ 90% trở lên) được tính là một, giữ lại cái điểm cao hơn. Sau đó `max_per_concept` giữ những cái tốt nhất trong phần còn lại.
- **Xem kết quả.** Nối `segs` và đúng bức ảnh đó vào [Segs Preview](#segs-preview-nh): mỗi instance được tô màu, có viền và gắn nhãn cùng điểm của nó.
- **Không tìm thấy gì** không phải lỗi: bạn nhận được không instance nào, một mask rỗng và `count` bằng 0.
- **Batch.** Mỗi ảnh được xử lý lần lượt và có danh sách instance riêng. Thanh tiến trình đếm theo concept, và bạn có thể hủy lần chạy giữa các concept và giữa các ảnh cận cảnh.
- **Ảnh rất rộng hoặc rất cao.** Trước hết SAM 3 nhìn cả bức ảnh bị nén vào một hình vuông. Với ảnh rộng hoặc cao hơn 4:1 (ảnh toàn cảnh), nó có thể bỏ sót vật nhỏ, và console cho biết điều đó. Hãy chia ảnh như vậy thành các ô bằng [Image Tile (NH)](./image-compose.md#image-tile-nh) rồi chạy từng ô.
- **Lỗi cài đặt** làm node dừng lại với thông báo nói rõ cần làm gì: thiếu gói, thiếu checkpoint, không có GPU CUDA. Xem [Khi có sự cố với SAM 3](#khi-có-sự-cố-với-sam-3).

> **Mẹo:** Hãy viết danh từ tiếng Anh đơn giản: `window`, `wine glass`, `red car`. Nếu một concept không tìm thấy gì, hãy thử từ khác (một tòa nhà có hình dạng lạ có thể đáp ứng `structure` hoặc `monument` thay vì `building`) hoặc hạ `threshold`.

> **Mẹo:** Hãy yêu cầu cả tổng thể lẫn các phần của nó (`person, hair, face, dress`), rồi tách chúng ra bằng [Segs Compose](#segs-compose-nh): `person - hair`.

> **Giấy phép:** SAM 3 thuộc SAM License của Meta. Xem [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3).

---

## Segs Select (NH)

<p align="center"><img src="../assets/nodes/NH_SegsSelect.svg" alt="Segs Select (NH) node" width="360"></p>

Giữ lại những instance bạn muốn, theo nhãn, điểm, kích thước và vị trí, theo thứ tự bạn chọn.

**Dùng khi** bạn chỉ cần một phần những gì SAM3 Segment tìm được: vật lớn nhất, người thứ hai từ trái sang, chỉ những kết quả chắc chắn, hay chỉ những ô cửa lớn.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc một node Segs khác. |
| `labels` | STRING | rỗng | Các nhãn cần giữ, cách nhau bằng dấu phẩy. `*` thay cho một chuỗi ký tự bất kỳ và `?` thay cho một ký tự (`*shirt`). Không phân biệt chữ hoa/thường, và `_` được coi là dấu cách (`red_car` tìm `red car`). Nhãn có dấu cách không cần ngoặc kép ở đây: `wine glass, red car`. Để trống thì giữ mọi nhãn. |
| `min_score` | FLOAT | `0.0` | Khoảng 0 đến 1. Loại các instance có điểm thấp hơn. |
| `min_area` | FLOAT | `0.0` | Khoảng 0 đến 1. Loại các instance nhỏ hơn tỉ lệ này của bức ảnh (`0.01` là 1%). |
| `max_area` | FLOAT | `1.0` | Khoảng 0 đến 1. Loại các instance lớn hơn tỉ lệ này của bức ảnh. |
| `sort_by` | COMBO | `score` | Thứ tự của các instance được giữ. Xem bảng bên dưới. |
| `start` | INT | `0` | Khoảng 0 đến 4096. Bỏ qua bấy nhiêu instance đầu của danh sách đã sắp xếp. `0` bắt đầu từ instance đầu tiên. |
| `count` | INT | `0` | Khoảng 0 đến 4096. Giữ bấy nhiêu instance kể từ `start`. `0` giữ tất cả. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `segs` | NH_SEGS | Các instance được giữ, theo thứ tự đã sắp xếp. |
| `mask` | MASK | Hợp của các instance được giữ, một mask cho mỗi ảnh. |
| `count` | INT | Số instance được giữ, tính trên cả batch. |

| `sort_by` | Thứ tự |
|---|---|
| `score` | Điểm cao nhất trước. |
| `area` | Lớn nhất trước. |
| `left_to_right`, `right_to_left` | Theo tâm ngang của khung. |
| `top_to_bottom`, `bottom_to_top` | Theo tâm dọc của khung. |
| `center_first` | Instance có tâm khung gần tâm bức ảnh nhất đứng trước. |
| `as_found` | Theo thứ tự SAM3 Segment tìm ra: lần lượt từng concept của bạn, điểm cao nhất trước trong mỗi concept. |

**Cách hoạt động**

- **Ba bước, theo thứ tự này:** các bộ lọc (`labels`, `min_score`, `min_area`, `max_area`), rồi sắp xếp, rồi cắt lấy một đoạn (`start` và `count`).
- **Theo từng ảnh.** Mỗi ảnh trong batch được xử lý riêng, nên `start` `1` bỏ qua instance đầu tiên của từng ảnh.
- **Diện tích** là tỉ lệ số điểm ảnh của bức ảnh mà mask phủ, không phải kích thước khung của nó. Vị trí dùng tâm của khung. Khi bằng nhau, thứ tự lúc tìm ra được giữ nguyên.
- Các instance được giữ vẫn theo thứ tự đã sắp, nên [Segs To Masks](#segs-to-masks-nh) đặt sau node này trả về các mask theo đúng thứ tự đó.

| Muốn có | Đặt |
|---|---|
| Người thứ hai từ trái sang | `labels` `person`, `sort_by` `left_to_right`, `start` `1`, `count` `1` |
| Vật lớn nhất | `sort_by` `area`, `count` `1` |
| Ba instance chắc chắn nhất | `sort_by` `score`, `count` `3` |
| Không có gì nhỏ hơn 1% bức ảnh | `min_area` `0.01` |
| Instance gần giữa ảnh nhất | `sort_by` `center_first`, `count` `1` |

---

## Segs Compose (NH)

<p align="center"><img src="../assets/nodes/NH_SegsCompose.svg" alt="Segs Compose (NH) node" width="360"></p>

Tạo một mask từ nhãn của các instance bằng một công thức nhỏ: hợp, trừ, giao và "mọi thứ trừ...".

**Dùng khi** bạn muốn có "phần nền", "người nhưng không có tóc", "bức tường nhưng không có cửa sổ", hoặc bất kỳ tổ hợp nào khác của những thứ SAM3 Segment tìm được.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc Segs Select. |
| `expression` | STRING | `all` | Công thức. Xem bên dưới. Ô này có nhiều dòng, và xuống dòng được coi là dấu cách. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Kết quả của công thức, một mask cho mỗi ảnh, có kích thước bằng ảnh. |

**Những gì bạn có thể viết**

| Viết | Nghĩa |
|---|---|
| `person` | Một nhãn: hợp của mọi instance mang nhãn đó. Không phân biệt chữ hoa/thường, `_` được coi là dấu cách, và `*`, `?` là ký tự đại diện (wildcard) (`*shirt`). |
| `"red car"` | Đặt nhãn có dấu cách hoặc ký tự toán tử trong ngoặc kép, `"` hoặc `'`, hoặc viết dấu cách thành `_` (`red_car`). Không có ngoặc kép, `red car` là lỗi. Ngoặc kép cũng biến một từ như `all` hay `image` thành nhãn thông thường. |
| `t-shirt` | Dấu gạch nối nằm giữa hai chữ cái hoặc chữ số thuộc về nhãn, nên `t-shirt` và `x-ray` không cần ngoặc kép. Dấu nháy đơn nằm giữa một từ cũng được (`women's`). Nhãn có dấu cách cần ngoặc kép, và dùng ngoặc kép đôi khi nhãn còn có dấu nháy đơn: `"women's dress"`. |
| `a \| b`, `a + b`, `a or b` | Hợp: mọi thứ thuộc a hoặc thuộc b. |
| `a - b` | a bỏ đi phần của b. Hãy đặt dấu cách hai bên dấu trừ: `person-hair` được đọc là một nhãn duy nhất. Khi cả `person` và `hair` đều có, node dừng với lỗi gợi ý `person - hair`; nếu không, nhãn lạ cho phần rỗng và console ghi tên nhãn đó. |
| `a & b`, `a and b` | Giao: chỉ nơi cả a và b cùng có. |
| `~a`, `!a`, `not a` | Mọi thứ trừ a: cả bức ảnh bỏ đi a. |
| `( … )` | Nhóm. |
| `all` | Mọi instance trong segs. |
| `image` | Cả bức ảnh. |
| `none` | Một mask rỗng. |

`~` được tính trước, rồi `&`, rồi `|` và `-` cùng cấp, từ trái sang phải: `a - b | c` nghĩa là `(a - b) | c`. Khi còn phân vân, hãy thêm dấu ngoặc.

| Biểu thức | Bạn nhận được |
|---|---|
| `person - hair` | Người nhưng bỏ phần tóc (hãy yêu cầu SAM3 Segment `person, hair`). |
| `~person` | Mọi thứ trừ người: phần nền. |
| `(wall \| ceiling) - window` | Tường và trần, khoét bỏ các ô cửa sổ. |
| `"wine glass" \| person` | Các ly rượu và người cùng lúc. |
| `all - person` | Mọi thứ đã tìm được trừ người. Khác với `~person`, nó không gồm phần nền mà không instance nào phủ tới. |

**Cách hoạt động**

- **Nhãn mà không instance nào có** cho một phần rỗng, và console báo điều đó (`no instance is labelled 'x'; those parts are empty`). Vì thế gõ sai `~persn` sẽ cho cả bức ảnh: hãy xem console khi mask trông lạ.
- **Mỗi ảnh** trong batch có mask riêng, dựng từ chính các instance của nó.
- **Công thức không đọc được** làm node dừng lại với thông báo nêu vị trí của chỗ sai, ví dụ `unexpected 'car' at position 5`.

---

## Segs To Masks (NH)

<p align="center"><img src="../assets/nodes/NH_SegsToMasks.svg" alt="Segs To Masks (NH) node" width="360"></p>

Đổi các instance thành mask thông thường: một mask cho mỗi instance, mask hợp của từng ảnh, và danh sách nhãn.

**Dùng khi** bạn cần mỗi instance là một mask riêng: để xử lý từng cái một, để lưu lại, hoặc để đọc nhãn của chúng.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc Segs Select. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `masks` | MASK | Một mask cho mỗi instance, ảnh này sang ảnh khác, theo thứ tự của `labels`. Khi hoàn toàn không có instance nào, đầu ra là một mask rỗng duy nhất, để các node phía sau không bị lỗi. |
| `union` | MASK | Một mask cho mỗi ảnh: hợp các instance của ảnh đó. |
| `labels` | STRING | Nhãn của từng instance, mỗi dòng một nhãn, theo thứ tự của `masks`. |
| `count` | INT | Số instance. Bằng 0 khi không có instance nào, dù lúc đó `masks` chứa một mask rỗng. |

> **Lưu ý:** Mỗi mask có kích thước bằng bức ảnh và 4 byte cho mỗi điểm ảnh: khoảng 145 MB trên ảnh 7360×4912. Khi các mask cần hơn khoảng một nửa dung lượng RAM còn trống của bạn (ví dụ hàng trăm instance trên ảnh 8K), node dừng lại với một thông báo thay vì làm đầy bộ nhớ. Hãy dùng [Segs Select](#segs-select-nh) trước để chỉ giữ những instance bạn cần, hoặc [Segs Compose](#segs-compose-nh) để lấy một mask duy nhất.

---

## Segs To Impact SEGS (NH)

<p align="center"><img src="../assets/nodes/NH_SegsToImpactSEGS.svg" alt="Segs To Impact SEGS (NH) node" width="360"></p>

Chuyển các instance của một ảnh sang `SEGS` của Impact Pack, để các detailer và công cụ SEGS của nó làm việc được với chúng.

**Dùng khi** bạn đã dùng Impact Pack và muốn các detailer của nó tinh chỉnh những gì SAM3 Segment tìm được. NH-Nodes không cần Impact Pack cho bất kỳ việc nào khác, nên hãy bỏ qua node này nếu bạn không dùng nó.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc Segs Select. |
| `crop_factor` | FLOAT | `3.0` | Khoảng 1 đến 10. Kích thước vùng mà detailer xử lý quanh mỗi instance, tính bằng bội số của khung của nó (cũng là `crop_factor` của Impact Pack). Vùng được dịch để nằm trong ảnh và luôn chứa trọn khung. |
| `batch_index` | INT | `0` | Khoảng 0 đến 4095. SEGS của Impact Pack chỉ chứa một ảnh, nên đầu vào này chọn ảnh nào trong batch được chuyển. Số vượt quá ảnh cuối làm node dừng lại với một thông báo. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `segs` | SEGS | `SEGS` của Impact Pack cho ảnh đó. Mỗi instance có khung, vùng cắt (crop region), mask, nhãn và điểm (là `confidence`). |

**Cách hoạt động**

- Đầu ra là kiểu riêng của Impact Pack, không phải `NH_SEGS`. NH-Nodes dựng nó với cùng các trường và không import Impact Pack, nên việc chuyển đổi không cần Impact Pack. Còn dùng kết quả thì cần.
- Muốn chuyển nhiều ảnh trong batch, hãy dùng mỗi `batch_index` một node.

---

## Segs Preview (NH)

<p align="center"><img src="../assets/nodes/NH_SegsPreview.svg" alt="Segs Preview (NH) node" width="360"></p>

Vẽ các instance lên ảnh của chúng: mỗi instance được tô màu riêng, có viền và gắn nhãn cùng điểm của nó.

**Dùng khi** bạn muốn xem SAM3 Segment đã tìm được gì, hoặc Segs Select và Segs Compose đang làm việc với cái gì. Hãy đặt node này sau SAM3 Segment để xem tất cả, hoặc sau Segs Select để chỉ xem phần đã chọn.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Bức ảnh mà segs được tìm ra từ đó: cùng số ảnh và cùng kích thước. Nếu không, node dừng lại và nói rõ. |
| `segs` | NH_SEGS | — | Các instance cần vẽ. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `preview` | IMAGE | Bức ảnh với các instance được vẽ lên, cùng kích thước. |

Màu sắc theo thứ tự của các instance, mỗi nhãn nằm phía trên điểm cao nhất của instance của nó, và cỡ chữ của nhãn theo kích thước bức ảnh. Ảnh 8K cho ra bản xem trước 8K.

---

## Workflow với SAM 3

Trong mỗi workflow dưới đây, hãy kiểm tra những gì SAM3 Segment tìm được trước khi chỉnh sửa bất cứ thứ gì. Nối đúng bức ảnh đó và `segs` vào [Segs Preview (NH)](#segs-preview-nh): mọi instance bạn muốn đều phải có viền và nhãn. Nếu thiếu một cái, hãy hạ `threshold` (ví dụ xuống 0.3) hoặc thử từ khác cho concept.

### Nền phía sau người

```text
Load Image → SAM3 Segment (concepts: person)
           → Segs Compose (expression: ~person)
           → Mask Morphology (NH)
           → chỉnh sửa phần nền
```

`~person` là cả bức ảnh trừ những người trong đó. Hãy thêm những thứ người đang cầm hoặc ngồi lên vào cả hai ô, ví dụ concept `person, wine glass, chair` và biểu thức `~(person | "wine glass" | chair)`. Dịch cạnh của mask vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh): mở rộng để phần chỉnh sửa phủ lên đường viền cũ, hoặc thu hẹp nếu phần chỉnh sửa lấn vào người. Sau đó dùng mask này để làm mờ, đổi ánh sáng, đổi màu hoặc inpaint phần nền.

### Vật lớn nhất, rồi inpaint

```text
Load Image → SAM3 Segment (concepts: person, dog, car)
           → Segs Select (sort_by: area, count: 1)
           → Mask Morphology (NH, mở rộng)
           → inpaint
```

Khi `labels` để trống, Segs Select chọn cái lớn nhất trong mọi thứ bạn yêu cầu. Hãy đặt `labels` (ví dụ `car`) để chọn cái lớn nhất của một loại, `start` `1` để lấy cái lớn thứ hai, và tăng `min_score` để một kết quả đáng ngờ không thể bị chọn. Hãy mở rộng mask thêm vài pixel để vùng inpaint phủ lên đường viền cũ.

### Kiến trúc: mặt tiền không có cửa sổ

```text
Load Image → SAM3 Segment (concepts: building, window, door)
           → Segs Compose (expression: building - (window | door))
           → đổi màu, đổi phong cách hoặc inpaint bức tường
```

`building` phủ cả mặt tiền cùng cửa sổ và cửa ra vào; công thức khoét chúng ra và chừa lại bức tường. Với một căn phòng, hãy yêu cầu `wall, ceiling, window` và dùng `(wall | ceiling) - window`. Nếu `building` không tìm thấy gì trên một tòa nhà có hình dạng lạ, hãy thử `structure` hoặc `monument` và sửa công thức cho khớp. Trên ảnh 4K đến 8K, hãy giữ `balanced`: mỗi cửa sổ được phân đoạn lại trên một ảnh cận cảnh của các điểm ảnh gốc. Đường viền của một mặt tiền chiếm trọn khung hình chỉ chính xác đến khoảng 1/300 bề ngang của nó (xem [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động)), nên hãy kiểm tra đường mái trong Segs Preview.

---

## Khi có sự cố với SAM 3

| Bạn thấy | Cần làm |
|---|---|
| Lỗi báo SAM 3 cần gói sam3 của Meta | Cài nó như trong [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3), rồi khởi động lại ComfyUI. |
| Lỗi báo gói sam3 đã cài nhưng không nạp được, hoặc thiếu một thành phần | Phiên bản đã cài không phải bản NH-Nodes hỗ trợ, hoặc thiếu một gói đi kèm. Hãy cài lại bằng hai dòng trong [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3). Thông báo cho biết nguyên nhân. |
| Lỗi báo không tìm thấy checkpoint | Tải `sam3.pt` như trong [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3), nhấn R trong ComfyUI và chọn tệp. Thông báo liệt kê các thư mục đã tìm. |
| Lỗi báo một tệp không phải checkpoint `.pt` của SAM 3 | Hãy dùng `sam3.pt`. Các tệp safetensors trên Hugging Face không dùng được với node này. |
| Lỗi báo không đọc được một tệp như checkpoint SAM 3, hoặc tệp đó không phải checkpoint SAM 3 | Tệp trong thư mục `sam3` là bản tải dở hoặc là `.pt` của model khác. Hãy tải lại `sam3.pt` như trong [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3). |
| Lỗi báo SAM 3 cần GPU NVIDIA có CUDA | ComfyUI đang chạy trên CPU hoặc trên loại GPU khác. SAM3 Segment không có chế độ CPU. |
| Lỗi báo bạn phải viết ít nhất một concept | `concepts` đang trống hoặc chỉ có dòng ghi chú. Hãy viết một từ như `person`. |
| `count` bằng 0, hoặc thiếu một instance | Hạ `threshold`, thử từ khác cho concept, và xem kết quả bằng Segs Preview. |
| Console báo đã giữ mask lượt đầu cho một số instance | Ảnh cận cảnh của chúng không khớp với lượt đầu, nên các instance đó giữ cạnh của lượt đầu. Không có gì sai. |
| Lỗi từ Segs Preview về kích thước hoặc số ảnh | Hãy nối đúng bức ảnh mà segs được tìm ra từ đó. |
| Lỗi từ Segs Compose có nêu một vị trí | Kiểm tra công thức tại vị trí đó. Đặt nhãn có dấu cách trong ngoặc kép: `"red car"`. |
| Mask của Segs Compose rỗng, và console báo không instance nào có một nhãn | Kiểm tra chính tả của nhãn. Đặt dấu cách quanh dấu trừ: `person - hair`, không phải `person-hair`. |
| Lỗi từ Segs To Masks báo các mask instance kích thước đầy đủ cần nhiều bộ nhớ hơn mức còn trống | Có quá nhiều instance so với kích thước bức ảnh. Dùng Segs Select để chỉ giữ những instance bạn cần, hoặc Segs Compose để lấy một mask duy nhất. |
| Console báo SAM 3 nhìn bức ảnh bị nén vào hình vuông và có thể bỏ sót vật nhỏ | Bức ảnh rộng hoặc cao hơn 4:1. Hãy chia nó thành các ô bằng Image Tile (NH) rồi chạy từng ô. |
| ComfyUI hết VRAM sau khi node chạy | Để `keep_model_loaded` tắt, để trọng số rời GPU sau mỗi lần chạy. |

---

[← Về trang hướng dẫn](./README.md)
