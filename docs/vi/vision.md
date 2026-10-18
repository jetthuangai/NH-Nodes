[← Về trang hướng dẫn](./README.md)

# Vision (phân đoạn ảnh)

Chín node cắt từng phần trên ảnh chụp. Ba node trong số đó phân tích người bằng các model hiểu về con người: trang phục với Garment Segment, khuôn mặt, tóc, cổ với Portrait Segment, còn Body Segment cho bạn gõ tên các bộ phận muốn lấy, như da, tóc, môi, răng, tay, chân và quần áo của mọi người trong ảnh. Mỗi node trả về một mask có cùng kích thước với ảnh đầu vào, một ảnh cắt nền có độ trong suốt, một ảnh xem trước mask và các bộ phận nó tìm được dưới dạng instance. Sáu node còn lại dùng SAM 3, model phân đoạn của Meta: bạn gõ thứ mình muốn vào SAM3 Segment (người, ly rượu, cửa sổ, bầu trời, hoặc một công thức như `person - bag` hay `background`) và nó trả về mask, đồng thời hiển thị ngay trên node. Nó cũng giữ lại mọi instance (từng đối tượng riêng lẻ) tìm được, và năm node Segs giúp chọn, ghép, xem trước và chuyển đổi chúng, kể cả cùng với các bộ phận mà ba node phân tích người tìm được. Bạn tìm thấy chúng trong **NH-Nodes → Vision**.

**Trên trang này**

- [Nên dùng node nào?](#nên-dùng-node-nào) — các node phân tích người hay SAM 3
- **Các node phân tích người: trang phục, chân dung và bộ phận cơ thể**
  - [Trước khi bắt đầu](#trước-khi-bắt-đầu) — model, bộ nhận diện khuôn mặt, lần chạy đầu tiên, giấy phép, bộ nhớ
  - [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh) — vì sao khuôn mặt và người nhỏ được xem cận cảnh, kể cả với Body Segment
  - [Tinh chỉnh mask](#tinh-chỉnh-mask) — các tùy chọn dùng chung của cả ba node
  - [Garment Segment (NH)](#garment-segment-nh) — mask trang phục thân trên, thân dưới, giày dép, mũ nón, hoặc riêng một chiếc đầm, túi, khăn…; hai model để chọn
  - [Portrait Segment (NH)](#portrait-segment-nh) — mask khuôn mặt, tóc, cổ và các bộ phận trên mặt
  - [Body Segment (NH)](#body-segment-nh) — gõ tên bộ phận muốn lấy: da, tóc, môi, răng, tay, chân, quần áo
- **Các node SAM 3: bất cứ thứ gì bạn gọi tên được**
  - [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3) — GPU, gói cài đặt, checkpoint, bộ nhớ, giấy phép
  - [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động) — `balanced` làm được gì trên ảnh 4K đến 8K, và giới hạn ở đâu
  - [Các instance: kiểu NH_SEGS](#các-instance-kiểu-nh_segs) — thứ được chuyển qua lại giữa các node, và node nào tạo ra nó
  - [Viết công thức](#viết-công-thức) — những gì `concepts`, `expression` và `regions` nhận: danh sách, `-`, `&`, `~`, `background`, từ tiếng Việt
  - [SAM3 Segment (NH)](#sam3-segment-nh) — gõ thứ bạn muốn, nhận mask của nó và xem ngay trên node
  - [Segs Select (NH)](#segs-select-nh) — giữ instance theo nhãn, điểm, kích thước và vị trí
  - [Segs Compose (NH)](#segs-compose-nh) — dựng mask từ nhãn bằng `|`, `-`, `&` và `~`, kể cả từ nhiều node cùng lúc
  - [Segs To Masks (NH)](#segs-to-masks-nh) — mỗi instance một mask, cùng mask hợp và danh sách nhãn
  - [Segs To Impact SEGS (NH)](#segs-to-impact-segs-nh) — đưa các instance cho detailer của Impact Pack
  - [Segs Preview (NH)](#segs-preview-nh) — vẽ các instance lên ảnh của chúng
  - [Workflow với SAM 3](#workflow-với-sam-3) — nền phía sau người, vật lớn nhất, mặt tiền không có cửa sổ
  - [Khi có sự cố với SAM 3](#khi-có-sự-cố-với-sam-3)

## Nên dùng node nào?

| Bạn muốn | Dùng | Cần gì |
|---|---|---|
| Trang phục của một người hoặc cả nhóm: áo, đầm, quần, giày, mũ, túi, khăn | [Garment Segment](#garment-segment-nh) | Một model SegFormer (256 MB) và một bộ nhận diện khuôn mặt nhỏ, đều tự tải ở lần chạy đầu. Chạy được trên CPU hoặc GPU. |
| Khuôn mặt, tóc, cổ và các bộ phận trên mặt của một ảnh chân dung, ảnh nửa người hoặc mọi khuôn mặt trong ảnh nhóm | [Portrait Segment](#portrait-segment-nh) | Giống như trên, với model 340 MB. |
| Da, tóc, môi, răng, tay, chân hoặc quần áo của những người trong ảnh, gọi bằng tên bộ phận (`skin`, `hair, face`, `clothes - shoes`, hay `da`, `tóc, môi`) | [Body Segment](#body-segment-nh) | Một model Sapiens2 (1.6 GB) và cùng bộ nhận diện khuôn mặt, đều tự tải ở lần chạy đầu, cùng PyTorch 2.4 trở lên. Nên chạy trên GPU: CPU rất chậm. |
| Bất cứ thứ gì bạn gọi tên được (người, đồ vật, bầu trời, tường, cửa sổ) dưới dạng mask từ một công thức ngắn (`person - bag`, `background`), hoặc từng instance riêng, kể cả trên ảnh 4K đến 8K | [SAM3 Segment](#sam3-segment-nh) và các node Segs | GPU NVIDIA, gói `sam3` của Meta và một checkpoint mà bạn tự cài và tự tải. Xem [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3). |

Bạn cũng có thể yêu cầu SAM3 Segment tìm `dress` hay `hair`. Các node phân tích người vẫn là lựa chọn dễ hơn cho những bộ phận cố định của chúng: chúng tự thiết lập, xử lý mọi người trong ảnh cùng một lúc, còn Garment Segment và Portrait Segment cũng chạy được không cần GPU.

**Chọn giữa các node phân tích người.** Garment Segment phân biệt được từng loại trang phục (áo với đầm, chân váy với quần) và còn tìm được túi, mũ, khăn, kính và trang sức. Body Segment gọi tên cơ thể: da, tóc, môi, răng, tay, chân và quần áo nói chung. Portrait Segment là node biết đến mắt, lông mày, mũi và tai. [Segs Compose](#segs-compose-nh) trộn những gì chúng tìm được, ví dụ `face - eyes - eyebrows` (khuôn mặt của cả Body Segment lẫn Portrait Segment, bỏ mắt và lông mày của Portrait Segment).

## Trước khi bắt đầu

Phần này nói về ba node phân tích người: Garment Segment, Portrait Segment và Body Segment. SAM3 Segment và các node Segs có phần riêng: [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3).

| Node và model | Kho Hugging Face | Dung lượng tải | Giấy phép |
|---|---|---|---|
| Garment Segment (NH), `model` `fashn` (mặc định): FASHN Human Parser, một SegFormer-B4 tinh chỉnh từ `nvidia/mit-b4`, với 18 lớp cho thời trang | `fashn-ai/fashn-human-parser` | khoảng 256 MB | NVIDIA Source Code License for SegFormer: chỉ để nghiên cứu hoặc đánh giá |
| Garment Segment (NH), `model` `fashionpedia`: SegFormer huấn luyện cho các lớp trang phục của Fashionpedia, model của 1.8.0 trở về trước | `1038lab/segformer_fashion` | khoảng 190 MB | Người đăng tải không khai báo, nhưng model được tinh chỉnh từ `nvidia/mit-b3`: hãy coi như `fashn` |
| Portrait Segment (NH): SegFormer tinh chỉnh (fine-tune) trên CelebAMask-HQ cho các bộ phận khuôn mặt | `jonathandinu/face-parsing` | khoảng 340 MB | Chỉ dành cho nghiên cứu và giáo dục phi thương mại |
| Body Segment (NH): Sapiens2 (Meta, 2026), với 28 lớp bộ phận cơ thể và trang phục, cộng với nền | `facebook/sapiens2-seg-0.4b`, `-0.8b` hoặc `-1b` | 1.6 GB (`0.4b`), 3.3 GB (`0.8b`) hoặc 5.9 GB (`1b`) | Sapiens2 License (Meta): miễn phí bản quyền, không có điều khoản cấm thương mại, nhưng có hạn chế về mục đích sử dụng |

**Lần chạy đầu tiên.** Khi chạy lần đầu, mỗi node tải model của nó từ Hugging Face, cùng với bộ nhận diện khuôn mặt mô tả bên dưới: với Garment Segment là model chọn trong `model`, với Body Segment là cỡ chọn trong `model_size`. Mỗi lần tải được cố định (pinned) vào một phiên bản xác định của kho, nên thay đổi về sau ở nguồn không thể làm đổi thứ bạn nhận được. Các tệp được lưu vào `ComfyUI/models/NH-Nodes/`, mỗi model một thư mục con (`fashn_human_parser`, `segformer_fashion`, `face_parsing`, `sapiens2-seg-<size>`). Nếu đã có sẵn một bản đầy đủ thì node dùng luôn bản đó: trong `models/RMBG/segformer_fashion` với model Fashionpedia, trong `models/face_parsing` hoặc `models/RMBG/segformer_face` với model khuôn mặt, hoặc trong bất kỳ thư mục nào bạn thêm bằng khóa `nh_nodes` trong `extra_model_paths.yaml` (xem ví dụ ở mục [Installation](../../README.md#installation)). Nếu quá trình tải bị gián đoạn, hãy chạy lại node: các tệp đã tải xong được giữ lại. Máy đang offline sẽ báo lỗi ở lần chạy đầu của một model cho đến khi model đó được tải về.

**Bộ nhận diện khuôn mặt.** Cả ba node còn dùng YuNet, một bộ nhận diện khuôn mặt nhỏ (khoảng 230 KB, giấy phép MIT, từ OpenCV Zoo), để tìm xem người đang ở đâu trước khi phân tích một bức ảnh. Nó được tải một lần, cố định vào một phiên bản xác định của `opencv/face_detection_yunet`, vào `ComfyUI/models/NH-Nodes/yunet/`. Nó chạy trên CPU bằng OpenCV mà NH-Nodes vốn đã cần, nên không tốn VRAM. Nếu đã có sẵn một bản trong thư mục con `yunet` của bất kỳ thư mục `nh_nodes` nào (`models/NH-Nodes/`, hoặc thư mục bạn thêm trong `extra_model_paths.yaml`) thì node dùng luôn bản đó; muốn chạy offline, hãy tự đặt `face_detection_yunet_2023mar.onnx` vào đó. Nếu không nạp được bộ nhận diện (offline ở lần chạy đầu, hoặc OpenCV quá cũ không chạy được nó), node in một cảnh báo trong console và chạy tiếp mà không có nó. Khi đó Portrait Segment phân tích cả bức ảnh, không có các ảnh cận cảnh nói ở [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh). Garment Segment và Body Segment vẫn xem cận cảnh những người mà lượt nhìn đầu tiên trên cả bức ảnh tìm thấy, nhưng có thể bỏ sót người mà lượt nhìn đầu không thấy. Hãy khởi động lại ComfyUI để thử lại.

**Bộ nhớ.** Model được đọc từ đĩa một lần và giữ trong bộ nhớ hệ thống. Nó chỉ chuyển sang GPU trong lúc node chạy, rồi trở về bộ nhớ hệ thống, nên không chiếm VRAM giữa các lần chạy. Trước khi chuyển sang GPU, node nhờ ComfyUI gỡ bớt các model của chính nó nếu VRAM đang eo hẹp. Body Segment chỉ giữ một cỡ Sapiens2 trong bộ nhớ mỗi lần: khi bạn chọn cỡ khác, cỡ cũ bị bỏ đi.

> **Lưu ý:** Nếu node báo nhãn không như mong đợi (unexpected labels), nghĩa là có một checkpoint khác nằm trong một thư mục mà node tìm. Hãy xóa bản đó, model đúng sẽ được tải về. Body Segment cũng báo tương tự khi một tệp không phải checkpoint Sapiens2 của cỡ bạn chọn.

> **Giấy phép:** Các model này đi kèm điều khoản giới hạn cách bạn được dùng chúng. Model đứng sau Portrait Segment chỉ được dùng cho nghiên cứu và giáo dục phi thương mại. Cả hai model của Garment Segment đều xây trên SegFormer của NVIDIA. FASHN Human Parser theo NVIDIA Source Code License for SegFormer; mục 3.3 của giấy phép này nói rằng "the Work" (phần mềm và các sản phẩm phái sinh của nó) "only may be used or intended for use non-commercially" (chỉ được dùng, hoặc chỉ được dự định dùng, một cách phi thương mại) và định nghĩa điều đó là "for research or evaluation purposes only" (chỉ cho mục đích nghiên cứu hoặc đánh giá): hẹp hơn nhiều so với "mọi việc dùng mà không thu tiền", đó là nghiên cứu hoặc đánh giá và không gì khác. Model Fashionpedia không khai báo giấy phép, nhưng được tinh chỉnh từ `nvidia/mit-b3`, nên nhiều khả năng cùng giấy phép đó cũng áp dụng. Sapiens2, đứng sau Body Segment, theo Sapiens2 License của Meta: miễn phí bản quyền, không có điều khoản cấm thương mại, nhưng cấm theo dõi (surveillance), xử lý sinh trắc học, nhận dạng người, deepfake và một số mục đích khác, đồng thời yêu cầu bạn tuân thủ luật về quyền riêng tư, dữ liệu sinh trắc học và kiểm soát thương mại. Mã nguồn của NH-Nodes thuộc giấy phép MIT. Bộ nhận diện khuôn mặt YuNet theo giấy phép MIT và không thay đổi điều nào ở trên. Hãy đọc [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) trước khi dùng bất kỳ node nào trong số này cho công việc có thu phí.

## Cách các node SegFormer nhìn một bức ảnh

Các model đứng sau những node này nhìn một bức ảnh nhỏ có kích thước cố định (rộng × cao): 512 × 512 với model khuôn mặt và model Fashionpedia, 384 × 576 với FASHN, 768 × 1024 với Sapiens2. Trên ảnh 6000 px, một khuôn mặt rộng 600 px chỉ còn rộng khoảng 50 px trong mắt model 512 px, nên các node không đưa cho model cả bức ảnh nguyên khối.

1. **Cả bức ảnh trước.** Ảnh lớn được thu nhỏ bằng cách lấy trung bình theo diện tích (area averaging), nhanh hơn phép resize riêng của model và giữ chi tiết nhỏ sạch sẽ, rồi mới được phân tích. Kết quả đầu tiên này được giữ lại ở mọi chỗ chưa có gì tốt hơn, nên những gì các bước sau bỏ sót vẫn giữ ít nhất kết quả này.
2. **Tìm người.** Bộ nhận diện khuôn mặt tìm khuôn mặt trên hai kích cỡ của bức ảnh. Garment Segment và Body Segment còn lấy thêm những người mà lượt phân tích đầu đã tìm thấy.
3. **Ảnh cận cảnh tinh chỉnh kết quả.** Mỗi khuôn mặt (Portrait Segment) hoặc mỗi nhóm người (Garment Segment, Body Segment) được phân tích lại trên một vùng cắt mà model thấy ở đủ độ phân giải của nó. Nhãn của vùng cắt được đưa vào kết quả đầu, nhưng không đưa dọc các cạnh cắt của vùng, nơi model thấy vật bị cắt cụt, nên không có cạnh thẳng nào lộ ra trong tóc hay quần áo. Mức độ tin vào một vùng cắt khác nhau tùy node, như các mục của từng node mô tả. Vùng cắt thấp hơn 128 px thì không được phân tích lại: nó sẽ bị phóng lên 4 đến 8 lần để vừa với đầu vào của model, nên phần lớn những gì model trả về là bịa ra, và nó hiện ra thành các đốm nhãn li ti rải rác (ví dụ một vật nhỏ cầm trên tay). Nhãn của lượt nhìn đầu được giữ nguyên ở đó.

Với ảnh dài hơn 4096 px, toàn bộ việc trên được làm trên một bản sao dài 4096 px rồi bản đồ nhãn được phóng lên đúng kích thước ảnh ở cuối: các model chỉ nhìn 384 đến 1024 px mỗi lần nên nhãn của chúng không chứa chi tiết mịn hơn, còn ghép các vùng cắt ở đủ kích thước 8K sẽ mất nhiều giây.

Body Segment chạy Sapiens2 chứ không phải SegFormer, và làm đúng ba bước trên với các vùng cắt có hình dạng 3:4 riêng của nó. Nó thấy mỗi vùng cắt ở 768 × 1024, nên mỗi người có khoảng 1024 px dành cho model dù ảnh lớn cỡ nào. Nó thêm một bước kiểm tra ngay từ đầu: khi không tìm thấy khuôn mặt nào, node coi như bức ảnh không có ai nếu Sapiens2 không chắc về phần lớn những gì nó gán là người, hoặc nếu một lượt nhìn vào bức ảnh lật gương không tìm thấy cùng những người đó (xem [Body Segment](#body-segment-nh)).

Mỗi node dùng điều này thế nào được nêu trong mục riêng của nó: [Garment Segment](#garment-segment-nh), [Portrait Segment](#portrait-segment-nh) và [Body Segment](#body-segment-nh).

## Tinh chỉnh mask

Garment Segment, Portrait Segment và Body Segment biến bản đồ nhãn của model thành mask, rồi tinh chỉnh theo thứ tự: `fill_holes`, `mask_expand`, `mask_blur`. Mask luôn có kích thước bằng ảnh đầu vào.

| Tùy chọn | Khoảng | Garment Segment | Portrait Segment | Body Segment | Tác dụng |
|---|---|---|---|---|---|
| `fill_holes` | bật / tắt | tắt | bật | tắt | Lấp các lỗ nằm trọn trong mask. Với trang phục, mặc định là tắt vì nếu bật, cánh tay vắt ngang thân (nằm lọt giữa vùng trang phục) sẽ bị lấp vào mask. Với chân dung thì bật, vì khuôn mặt vẫn đặc ở nơi kính, hoa tai hay tóc cắt vào nó. |
| `mask_expand` | -64 đến 64 | `0` | `0` | `0` | Mở rộng (số dương) hoặc thu hẹp (số âm) mask đúng bấy nhiêu pixel. |
| `mask_blur` | 0 đến 64 | `0` | `0` | `0` | Làm mềm (feather) cạnh khoảng bấy nhiêu pixel. |
| `device` | `cuda`, `cpu` | `cuda` | `cuda` | `cuda` | Nơi model chạy. Không có CUDA thì node chuyển sang CPU và in một ghi chú một lần trong console. |

Ba đầu ra dưới đây dùng chung cho cả ba node. Mỗi node còn trả về `segs` (các bộ phận nó tìm được dưới dạng instance, mô tả trong mục riêng của nó), và Body Segment thêm một `report`.

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask với giá trị từ 0 đến 1. Cạnh mềm khi `mask_blur` lớn hơn 0. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. Màu bên ngoài mask được giữ nguyên. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh RGB màu xám, dùng để xem trước hoặc cho các node cần `IMAGE`. |

Batch (lô ảnh) đều dùng được: từng ảnh được phân đoạn lần lượt và các đầu ra giữ nguyên thứ tự batch.

---

## Garment Segment (NH)

<p align="center"><img src="../assets/nodes/NH_GarmentSegment.svg" alt="Garment Segment (NH) node" width="360"></p>

Tạo mask cho một nhóm trang phục, một món riêng (đầm, túi, khăn…), hoặc tất cả các nhóm cùng lúc, trên những người trong ảnh. Hai model làm việc này: `fashn` (mặc định) và `fashionpedia`.

**Dùng khi** bạn muốn có mask sạch của áo, đầm, quần, giày, mũ hoặc túi để inpaint, đổi màu hoặc dùng trong workflow thử đồ.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Ảnh chụp một người hoặc một nhóm người. |
| `part` | COMBO | `upper_garment` | Một nhóm: `upper_garment`, `lower_garment`, `footwear`, `headwear` hoặc `all_garments`; hoặc một món riêng: `dress`, `bag`, `belt`, `scarf`, `glasses` hoặc `jewelry`. Xem bảng bên dưới. |
| `attach_parts` | BOOLEAN | `true` | Chỉ với `fashionpedia`: gộp túi, khóa kéo, nơ, bèo nhún và các chi tiết tương tự chạm vào trang phục đã chọn. Không có tác dụng với `fashn`, vì model này tính chúng là một phần của món đồ. |
| `mask_expand` | INT | `0` | Khoảng -64 đến 64. Mở rộng (+) hoặc thu hẹp (-) mask N pixel. |
| `mask_blur` | INT | `0` | Khoảng 0 đến 64. Làm mềm cạnh mask N pixel. |
| `fill_holes` | BOOLEAN | `false` | Lấp các lỗ khép kín. Mặc định tắt: nếu bật, cánh tay vắt ngang thân sẽ bị lấp vào mask. |
| `device` | COMBO | `cuda` | `cuda` hoặc `cpu`. |
| `model` | COMBO | `fashn` | Tùy chọn (optional), để các workflow lưu từ trước khi có nó vẫn nạp được: chúng chạy `fashn`. `fashn` hoặc `fashionpedia`. Xem bảng bên dưới. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask của phần đã chọn. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh xem trước RGB màu xám. |
| `segs` | NH_SEGS | Mỗi bộ phận model tìm được là một instance, bất kể `part` là gì, dùng cho [Segs Compose](#segs-compose-nh) và các node Segs khác. Với `fashn`: face, hair, top, dress, skirt, pants, belt, bag, hat, scarf, glasses, arms, hands, legs, feet, torso, jewelry. Với `fashionpedia`: tên các lớp của nó (shirt, top, sweater, …, sleeve, pocket, …). Một instance phủ bộ phận của nó trên mọi người trong ảnh. |

**Chọn `model` nào**

| `model` | Là gì | Giỏi ở | Yếu ở |
|---|---|---|---|
| `fashn` (mặc định) | FASHN Human Parser, một SegFormer-B4 cho thời trang với 18 lớp, 256 MB | Áo, đầm, chân váy, quần, túi, mũ, khăn, kính và trang sức, với đường viền sạch. | Không có lớp giày, tất hay băng đô. Món đồ nhạt màu, gần với màu da. |
| `fashionpedia` | `1038lab/segformer_fashion`, model của 1.8.0 trở về trước, 190 MB | Giày, tất, băng đô và các chi tiết nhỏ (tay áo, cổ áo, túi áo). | Nguyên cả món đồ: bỏ sót một số chiếc đầm và áo. Không có lớp trang sức. |

**Mỗi `part` gồm những gì**

| `part` | Với `fashn` | Với `fashionpedia` |
|---|---|---|
| `upper_garment` | Áo, đầm và khăn quàng. | Áo sơ mi và áo kiểu, áo thun và áo nỉ, áo len, cardigan, áo khoác, áo gile, áo khoác dài (coat), đầm, jumpsuit, áo choàng (cape), cà vạt, khăn quàng, mũ trùm áo, cổ áo, ve áo, cầu vai, tay áo và đường viền cổ. |
| `lower_garment` | Chân váy, quần dài và thắt lưng. | Quần dài, quần short, váy, thắt lưng, khóa thắt lưng, bó chân (leg warmer), quần tất và tất dài. Không gồm giày. |
| `footwear` | Lớp "feet" của FASHN: bàn chân cùng thứ nó đang mang, nên chân trần cũng được lấy. | Giày và tất. |
| `headwear` | Mũ. Không có lớp băng đô. | Mũ, băng đô, khăn đội đầu và phụ kiện tóc. |
| `all_garments` | Bốn nhóm trên trong cùng một mask, gồm cả chân trần. | Bốn nhóm trên trong cùng một mask. |
| `dress` | Đầm. | Đầm và jumpsuit. |
| `bag` | Túi. | Túi xách và ví. |
| `belt` | Thắt lưng. | Thắt lưng và khóa thắt lưng. |
| `scarf` | Khăn quàng. | Khăn quàng. |
| `glasses` | Kính. | Kính. |
| `jewelry` | Trang sức. | Mask rỗng: model không có lớp trang sức, và console cho biết điều đó. Hãy chọn `fashn`. |

**Cách hoạt động**

- **Đầm và jumpsuit** thuộc `upper_garment`, cùng với các loại áo. `dress` chỉ lấy riêng các chiếc đầm. Muốn mask áo và quần cùng lúc, hoặc đầm và giày, hãy chọn `all_garments`.
- **Giày dép với `fashn`** là lớp "feet" của FASHN, không phải giày: bàn chân cùng thứ nó đang mang. Chân trần cũng nằm trong `footwear` và cả trong `all_garments`.
- **Chi tiết đi kèm (attached parts)** (chỉ `fashionpedia`) gồm túi (pocket), khóa kéo, họa tiết đính (appliqué), hạt cườm, nơ, hoa, tua rua, ruy băng, đinh tán, bèo nhún, kim sa (sequin) và tua (tassel). Khi bật `attach_parts`, một chi tiết chỉ được thêm vào khi nó chạm vào trang phục đã chọn (trong vòng vài pixel). Túi trên áo khoác sẽ nhập vào mask `upper_garment`, còn chiếc nơ trên một chiếc túi xách thì không. `fashn` không có các lớp này: nó tính túi áo, khóa kéo và miếng vá là một phần của món đồ.
- **Không lấy được qua `part`:** với `fashn` là khuôn mặt, tóc, cánh tay, bàn tay, chân và thân; với `fashionpedia` là găng tay, đồng hồ và ô. Chúng nằm trong `segs`.
- **Ảnh cận cảnh của người.** Node phân tích cả bức ảnh trước và giữ kết quả đó. Sau đó nó tìm những người (trong lượt phân tích ấy và bằng bộ nhận diện khuôn mặt) và cắt các vùng có đúng hình dạng đầu vào của model (vuông với `fashionpedia`, 2:3 với `fashn`) quanh từng nhóm người: một hàng các vùng cắt chồng lên nhau cho nhóm rộng, một vùng cho một người hoặc một nhóm gọn. Mỗi vùng được phân tích ở đủ độ phân giải của model, và vùng được nới rộng khi người bị tràn qua một cạnh của nó. Việc này giúp những người nhỏ trong một bức ảnh lớn. Xem [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh).
- **Vùng cắt chỉ tinh chỉnh.** Nhãn của nó chỉ thay kết quả đầu ở những chỗ model chắc hơn, cách xa cạnh cắt của vùng, và chỉ với các phần nằm trên những người đã tìm thấy. Không có gì bị bịa ra trên một phông trống, và không có cạnh thẳng nào của vùng cắt xẻ qua một món trang phục.

<p align="center"><img src="../assets/examples/garment-segment.jpg" alt="Mask của Garment Segment phủ lên một ảnh mẫu" width="720"><br><em>Garment Segment trên một ảnh mẫu: các vùng tô màu là mask của phần đã chọn, từ model fashn (mặc định) và từ model fashionpedia (màu hổ phách: các chi tiết mà attach_parts đã gộp vào), cùng ảnh cắt nền làm từ mask của fashn.</em></p>

**Ví dụ**

```text
Load Image → Garment Segment (NH, upper_garment) → mask → Mask Morphology (NH) → inpaint
```

Hãy mở rộng mask thêm vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh) để trang phục được inpaint có thể phủ lên đường viền cũ.

> **Lưu ý:** `fashn` có thể bỏ sót món đồ nhạt màu gần với màu da: một chiếc áo sát nách màu hồng nhạt chỉ được tìm thấy một phần. Lỗi nằm ở chính model, không phải ở các vùng cắt, và `top` hay `clothes` của Body Segment lấy được món đó tốt hơn. Nó cũng có thể gắn nhãn áo cho những thứ cầm sát người, ví dụ ly champagne. `fashionpedia` cũng bỏ sót một số áo, ví dụ chiếc áo cánh nằm dưới một chiếc mũ lớn.

> **Giấy phép:** cả hai model trang phục chỉ dùng cho nghiên cứu hoặc đánh giá. FASHN Human Parser theo NVIDIA Source Code License for SegFormer, còn model Fashionpedia, không khai báo giấy phép, được tinh chỉnh từ `mit-b3` của NVIDIA. Xem [Trước khi bắt đầu](#trước-khi-bắt-đầu).

---

## Portrait Segment (NH)

<p align="center"><img src="../assets/nodes/NH_PortraitSegment.svg" alt="Portrait Segment (NH) node" width="360"></p>

Tạo mask cho khuôn mặt, tóc, cổ và các bộ phận trên mặt của một ảnh chân dung, hoặc của mọi khuôn mặt trong ảnh nhóm. Bạn bật những vùng muốn lấy.

**Dùng khi** bạn muốn chỉnh sửa hoặc đổi màu tóc, xử lý da và các bộ phận trên mặt, hoặc bảo vệ khuôn mặt trong khi chỉnh phần còn lại của ảnh.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Một ảnh chân dung, ảnh nửa người hoặc ảnh nhóm. |
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
| `mask` | MASK | Mask gộp của mọi vùng đã bật, trên mọi khuôn mặt tìm thấy. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh xem trước RGB màu xám. |
| `segs` | NH_SEGS | Mỗi bộ phận model tìm được là một instance, bất kể bạn đã bật vùng nào: skin, nose, glasses, eyes, eyebrows, ears, mouth (bên trong miệng), lips, hair, hat, earrings, necklace, neck, clothes. Một instance phủ bộ phận của nó trên mọi người trong ảnh. Dùng cho [Segs Compose](#segs-compose-nh) và các node Segs khác. |

**Cách hoạt động**

- Các vùng đã bật được gộp thành một mask. Không bật vùng nào thì mask rỗng.
- `face` đã bao gồm mắt, lông mày, mũi, môi và bên trong miệng. Các tùy chọn đó chỉ làm kết quả thay đổi khi `face` tắt.
- Mũ, hoa tai, dây chuyền và quần áo có trong model nhưng node không có công tắc cho chúng. Chúng nằm trong `segs`, nơi [Segs Compose](#segs-compose-nh) dùng được.
- **Cả bức ảnh trước.** Bức ảnh được phân tích dưới dạng một hình vuông có thêm viền (giữ nguyên tỉ lệ, phần còn lại được lấp bằng màu xám) thay vì bị nén vào hình vuông của model. Kết quả này được giữ lại ở mọi chỗ mà các vùng cắt bên dưới không chạm tới, nên một bức ảnh không tìm thấy khuôn mặt nào, hoặc không có bộ nhận diện, giữ nó làm kết quả. Khi đã tìm thấy khuôn mặt, các phần của mặt (da, mắt, lông mày, mũi, môi, tai) mà kết quả này cho thấy ở xa mọi khuôn mặt được bỏ đi vì là nhầm lẫn, chẳng hạn một vệt phản chiếu hay một hoa văn; tóc, cổ và quần áo của nó vẫn được giữ.
- **Khuôn mặt.** Bộ nhận diện khuôn mặt tìm các khuôn mặt (tối đa 32, lấy những khuôn mặt lớn nhất). Khi không tìm thấy khuôn mặt nào, bức ảnh được thêm một lề xám và tìm thêm một lần nữa, cách này tìm được khuôn mặt bị khung hình cắt. Sau đó model phân tích (parser) kiểm tra từng khuôn mặt trên một vùng cắt sát: bàn tay hay một họa tiết mà bộ nhận diện tưởng là khuôn mặt bị loại và không làm thay đổi gì.
- **Hai vùng cắt cho mỗi khuôn mặt.** Một vùng cắt rộng cho thấy trọn tóc và cổ. Một vùng cắt sát, đóng khung giống những bức ảnh model đã học, cho mắt, lông mày, mũi và môi. Vùng rộng tinh chỉnh phần nằm trong nó, cách xa các cạnh cắt của nó, còn vùng sát chỉ quyết định bên trong một hình elip quanh khuôn mặt, nên không có cạnh cắt thẳng nào lộ ra trong mái tóc dài. Nơi các vùng cắt của hai khuôn mặt chồng nhau, mỗi điểm ảnh thuộc về khuôn mặt gần hơn. Vì vậy trong ảnh nhóm, mask chứa các vùng đã bật của mọi người.
- **Khuôn mặt lớn.** Khuôn mặt đã chiếm một phần lớn khung hình thì được giao cho lượt phân tích cả bức ảnh, vì lượt đó đã thấy nó đủ lớn. Xem [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh).

<p align="center"><img src="../assets/examples/portrait-segment.jpg" alt="Mask của Portrait Segment phủ lên một ảnh mẫu" width="720"><br><em>Portrait Segment trên một ảnh mẫu: các vùng tô màu là mask của những vùng đã bật.</em></p>

**Ví dụ: chỉ lấy tóc**

```text
Load Image → Portrait Segment (NH) → mask → đổi màu hoặc inpaint
```

Tắt mọi vùng trừ `hair`. Với các giá trị mặc định, bạn có khuôn mặt, tai, cổ và tóc cùng trong một mask.

> **Mẹo:** Model được tinh chỉnh trên CelebAMask-HQ, một bộ ảnh chụp cận mặt, nên node phân tích mỗi khuôn mặt tìm được trên một vùng cắt cận cảnh. Một khuôn mặt nhỏ trong ảnh lớn, một ảnh nửa người hay một ảnh nhóm không cần cắt (crop) trước. Khuôn mặt mà bộ nhận diện bỏ sót (rất nhỏ, bị che hoặc quay đi) giữ kết quả của lượt phân tích cả bức ảnh, thô hơn: hãy tự cắt quanh nó khi cần chi tiết mịn nhất.

> **Giấy phép:** model face-parsing chỉ dành cho nghiên cứu và giáo dục phi thương mại. Xem [Trước khi bắt đầu](#trước-khi-bắt-đầu).

---

## Body Segment (NH)

<p align="center"><img src="../assets/nodes/NH_BodySegment.svg" alt="Body Segment (NH) node" width="360"></p>

Bạn gõ tên các bộ phận cơ thể muốn lấy (`skin`, `hair, face`, `clothes - shoes`, hoặc tiếng Việt `da`, `tóc, môi`) và nhận mask của chúng. Model là Sapiens2 (Meta, 2026). Nó gán cho từng điểm ảnh của mỗi người một trong 28 bộ phận, hoặc là nền: khuôn mặt và cổ, tóc, thân, cánh tay trên và dưới, bàn tay, đùi và cẳng chân, bàn chân, môi, răng, lưỡi, kính, áo và quần (trang phục thân trên và thân dưới), giày, tất và "apparel" (phụ kiện). Node đổi những từ bạn gõ thành một mask và hiển thị nó ngay trên node.

**Dùng khi** bạn muốn có mask da, tóc, môi, răng, bàn tay hay quần áo của những người trong ảnh chỉ bằng một node, không phải tô vẽ gì: để retouch da, đổi màu tóc hay môi, bảo vệ bàn tay, hoặc thay trang phục. Garment Segment hợp hơn khi cần riêng một món đồ (đầm, túi, khăn), Portrait Segment hợp hơn khi cần mắt, lông mày, mũi và tai, còn SAM3 Segment dùng cho mọi thứ không phải người.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Ảnh có người. Mọi ảnh trong batch được xử lý lần lượt. |
| `regions` | STRING | `skin` | Các bộ phận bạn muốn, gọi bằng tên, viết dưới dạng công thức (xem [Viết công thức](#viết-công-thức)): `skin`, `hair, face`, `clothes - shoes`, `~person`, `background`. Dấu phẩy hoặc xuống dòng để liệt kê nhiều thứ. Các từ nằm trong các bảng bên dưới, bằng tiếng Anh và tiếng Việt. Hãy viết ít nhất một từ. |
| `model_size` | COMBO | `0.4b` | `0.4b`, `0.8b` hoặc `1b`. Các cỡ lớn hơn chậm hơn và cần nhiều VRAM hơn. Xem "Cỡ model, bộ nhớ và tốc độ" bên dưới. |
| `mask_expand` | INT | `0` | Khoảng -64 đến 64. Mở rộng (+) hoặc thu hẹp (-) mask N pixel. |
| `mask_blur` | INT | `0` | Khoảng 0 đến 64. Làm mềm cạnh mask N pixel. |
| `fill_holes` | BOOLEAN | `false` | Lấp các lỗ khép kín. |
| `device` | COMBO | `cuda` | `cuda` hoặc `cpu`. CPU vẫn chạy được nhưng rất chậm. |
| `show_preview` | BOOLEAN | `true` | Tùy chọn (optional). Hiển thị kết quả ngay trên node: mask sáng và phần còn lại bị làm tối, các bộ phận mà mask giữ lại có viền và gắn nhãn, và các dòng báo cáo bên dưới. Tối đa 1536 px ở cạnh dài, và chỉ 16 ảnh đầu của một batch. Tắt: không vẽ gì, và node vẫn chạy (xem "Cách hoạt động" bên dưới). |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Mask của công thức, một mask cho mỗi ảnh, có kích thước bằng ảnh. |
| `cutout_rgba` | IMAGE | Ảnh đầu vào với mask làm kênh alpha. |
| `mask_image` | IMAGE | Mask dưới dạng ảnh xem trước RGB màu xám. |
| `segs` | NH_SEGS | Mỗi bộ phận tìm được là một instance, bất kể `regions` viết gì: hair, face, torso, arms, hands, legs, feet, lips, teeth, tongue, glasses, top, bottom, shoes, socks, accessories. Một instance phủ bộ phận của nó trên mọi người trong ảnh. Dùng cho [Segs Compose](#segs-compose-nh) và các node Segs khác. |
| `report` | STRING | Mỗi từ trong `regions` một dòng, kèm tỉ lệ của nó trong bức ảnh ("under 0.1%" khi tỉ lệ rất nhỏ, "not in the picture" khi không có, "of the pictures" với một batch). Tiếp theo là các ghi chú khi một từ mang nghĩa khác trong Sapiens2 so với cách nói thường ngày. |

**Các từ bạn có thể gõ**

Từ bạn gõ là tiếng Anh, hoặc những từ tiếng Việt thông dụng trong danh sách dùng chung. Không phân biệt chữ hoa/thường, số ít và số nhiều là cùng một từ (`lower leg` hay `lower legs`, `accessory` hay `accessories`), và nhiều từ liền nhau là một nhãn (`upper arm` không cần ngoặc kép). Các bảng dưới đây theo `REGIONS` và `WHOLE_LIMBS` trong `nodes/vision/body_segment_nodes.py`, nơi có danh sách đầy đủ.

| Viết | Bạn nhận được |
|---|---|
| `skin`, `body skin` | Da của khuôn mặt và cổ, thân, cánh tay, bàn tay, chân và bàn chân. Không gồm môi, răng, lưỡi hay tóc. |
| `face`, `neck`, `face skin` | Khuôn mặt và cổ cùng nhau: một lớp duy nhất, kèm cả mắt, lông mày và râu. |
| `head` | Khuôn mặt và tóc. |
| `hair` | Tóc. |
| `arm`, `arms`, `upper arm`, `upper arms`, `lower arm`, `lower arms`, `forearm`, `forearms` | Cánh tay, không gồm bàn tay. `upper arm` là phần phía trên khuỷu tay (bắp tay), `lower arm` và `forearm` là phần phía dưới (cẳng tay). |
| `hand`, `hands` | Bàn tay. |
| `leg`, `legs`, `thigh`, `thighs`, `upper legs`, `calf`, `calves`, `lower legs` | Chân, không gồm bàn chân. `thigh` và `upper legs` là phần phía trên đầu gối (đùi), `calf`, `calves` và `lower legs` là phần phía dưới (cẳng chân). |
| `foot`, `feet` | Bàn chân. |
| `torso`, `chest`, `belly` | Thân người: chỉ là một lớp, nên ba từ này cho cùng kết quả. |
| `lips`, `lip`, `upper lip`, `lower lip` | Môi (cả hai, hoặc một bên). |
| `teeth`, `tooth`, `upper teeth`, `lower teeth` | Răng (tất cả, hoặc một hàm). |
| `tongue` | Lưỡi. |
| `mouth` | Môi, răng và lưỡi cùng nhau. |
| `glasses`, `eyeglasses`, `sunglasses` | Kính. |
| `top`, `upper clothes`, `shirt`, `t-shirt`, `blouse`, `jacket`, `suit jacket`, `sweater`, `hoodie`, `coat`, `vest`, `bra`, `dress`, `wedding dress`, `ao dai` | Trang phục thân trên. Tất cả các từ này là cùng một lớp. |
| `bottom`, `lower clothes`, `pants`, `trousers`, `jeans`, `shorts`, `skirt` | Trang phục thân dưới. Tất cả các từ này là cùng một lớp. |
| `shoe`, `shoes`, `high heels`, `sandal`, `sandals`, `sneakers`, `boots` | Giày. |
| `sock`, `socks` | Tất. |
| `accessories`, `apparel` | Lớp phụ kiện duy nhất của model: mũ, túi và những thứ tương tự. |
| `clothes`, `clothing`, `outfit` | Trang phục thân trên và thân dưới, giày, tất và phụ kiện. |
| `person`, `body` | Mọi thứ model gán là người. |
| `left hand`, `right hand`, `left arm`, `right arm`, `left leg`, `right leg`, `left foot`, `right foot`, `left shoe`, `right shoe` | Chỉ một bên. Xem bên dưới về bên nào. |

| Tiếng Việt | Bạn nhận được |
|---|---|
| `da` | Da (`skin`). |
| `tóc` | Tóc. |
| `mặt`, `da mặt`, `cổ` | Khuôn mặt và cổ, một lớp duy nhất. |
| `đầu` | Đầu: khuôn mặt và tóc. |
| `môi`, `răng`, `lưỡi`, `miệng` | Môi, răng, lưỡi, và miệng (cả ba cùng nhau). |
| `tay`, `đôi tay`, `hai tay` | Cả cánh tay kèm bàn tay, đúng như cách người Việt hiểu chữ "tay". |
| `tay trái`, `tay phải` | Cả cánh tay trái hoặc phải kèm bàn tay của nó. |
| `cánh tay` | Cánh tay, không kèm bàn tay. |
| `bắp tay`, `cẳng tay` | Bắp tay (phần trên), cẳng tay (phần dưới). |
| `bàn tay` | Bàn tay. |
| `chân`, `đôi chân`, `hai chân` | Cả cái chân kèm bàn chân. |
| `chân trái`, `chân phải` | Cả cái chân trái hoặc phải kèm bàn chân của nó. |
| `đùi`, `bắp đùi` | Đùi. |
| `bắp chân`, `cẳng chân` | Cẳng chân, phần phía dưới đầu gối. |
| `bàn chân` | Bàn chân. |
| `ngực`, `bụng` | Thân người. |
| `áo`, `áo sơ mi`, `áo thun`, `áo khoác`, `áo len` | Trang phục thân trên. |
| `váy`, `đầm`, `váy cưới`, `áo dài` | Một chiếc đầm, mà model gán là trang phục thân trên. |
| `quần`, `quần jean`, `quần short` | Trang phục thân dưới. |
| `chân váy` | Chân váy: trang phục thân dưới. |
| `giày`, `giày cao gót`, `dép` | Giày. |
| `tất`, `vớ` | Tất. |
| `kính`, `kính mát` | Kính. |
| `quần áo`, `trang phục` | Quần áo nói chung. |
| `người`, `cơ thể` | Cả người. |
| `nền` | Từ khóa `background`: mọi thứ nằm ngoài người. |

Muốn chỉ lấy một bên, hãy viết `left hand`, `right foot`… bằng tiếng Anh, hoặc `tay trái`, `tay phải`, `chân trái` và `chân phải` bằng tiếng Việt. Màu sắc hay kích cỡ đứng sau một từ tiếng Việt (`áo đỏ`) không được hiểu ở đây: Body Segment gọi tên bộ phận, không tìm theo màu.

**Các từ mang nghĩa gì**

Sapiens2 có ít lớp hơn số từ người ta hay dùng, nên một số từ dùng chung một lớp. Báo cáo thêm ghi chú khi bạn dùng một trong những từ này. Các ghi chú về lớp face, về đầm và về chân váy chỉ xuất hiện khi bức ảnh có phần nào của vùng đó.

- **Trái và phải** là bên của chính người trong ảnh: `left hand` nằm ở bên phải bức ảnh khi người đó nhìn thẳng vào ống kính.
- **`face` là một lớp duy nhất**, gồm cả cổ, mắt, lông mày và râu, và `skin` bao gồm nó. Vì vậy `skin - face` cũng bỏ luôn cổ, `face - neck` rỗng, và mask da không có lỗ ở chỗ mắt. Muốn mask khuôn mặt không có mắt và lông mày, hãy dùng Portrait Segment, hoặc lấy `eyes` và `eyebrows` của nó trừ đi bằng Segs Compose (xem bên dưới).
- **Đầm được tính là áo.** Sapiens2 chỉ có một lớp cho mọi thứ mặc ở phần thân trên: `dress - top` rỗng, và phần chân váy của một chiếc đầm được tính là `top`, không phải `bottom`. `skirt` chỉ tìm chiếc chân váy rời.
- **`skin` không có môi, răng và lưỡi.** Chúng là các lớp riêng, nên `skin - lips` không đổi gì, còn `skin + lips` là da kèm cả môi.
- **`clothes` gồm cả phụ kiện** (túi, mũ), giày và tất.
- **`tay` và `chân` là cả chi**, gồm cả bàn tay và bàn chân. `arm` và `leg` là chi không kèm chúng, nên `tay - cánh tay` chỉ còn lại bàn tay.

**Những từ bị từ chối**

Nếu một từ không ứng với vùng nào mà Body Segment có, node dừng lại và báo lỗi, trước khi nạp model; lỗi cho biết nên viết gì thay thế:

| Bạn viết | Nên viết |
|---|---|
| `eyes`, `eyebrows`, `nose`, `ears` (cũng như `mắt`, `mũi`, `tai`) | Chúng là một phần của `face` ở đây. Portrait Segment mask từng bộ phận một. |
| `beard`, `mustache` (cũng như `râu`, `ria mép`) | Râu là một phần của `face` ở đây, và không parser nào của NH-Nodes có lớp râu. SAM3 Segment có thể thử `beard`. |
| `hat`, `bag`, `scarf`, `belt`, `jewelry`, `necklace`, `watch`, `gloves`, `tie` (cũng như `mũ`, `túi`, `khăn`) | Sapiens2 chỉ có một lớp cho phụ kiện: `accessories`. Muốn lấy riêng một món, hãy dùng Garment Segment hoặc SAM3 Segment. |
| `shoulders` | Chúng là một phần của `torso` và `upper arms`. |
| `fingers` | Chúng là một phần của `hands`. |
| `skin and hair`, `skin or hair`, `not skin` | `and`, `or` và `not` không phải toán tử ở đây: hãy viết `skin + hair` (hoặc `skin, hair`), `skin - hair` và `~skin`. |
| Ký tự đại diện (`*`, `?`) | Không được phép: hãy viết tên vùng. |
| Một từ tiếng Việt không có trong danh sách | Hãy viết bằng tiếng Anh, hoặc dùng `da`, `tóc`, `mặt`, `môi`, `răng`, `tay`, `chân`, `áo`, `quần`, `giày`. |
| Ô trống | Hãy viết ít nhất một vùng, ví dụ `skin`. |

**Cách hoạt động**

- **Người, xem cận cảnh.** Node phân tích cả bức ảnh trước, tìm người (trong lượt phân tích đó và bằng bộ nhận diện khuôn mặt), rồi phân tích các vùng cắt có hình dạng 3:4 của model quanh từng người hoặc từng nhóm ở 1024 × 768, nên mỗi người có khoảng 1024 px dành cho model dù ảnh lớn cỡ nào. Như ở Garment Segment, vùng cắt chỉ tinh chỉnh kết quả đầu, và với ảnh dài hơn 4096 px, việc này được làm trên một bản sao 4096 px rồi phóng lên ở cuối. Xem [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh).
- **Công thức.** Mỗi từ chọn một số lớp của model, và công thức kết hợp chúng như trong [Viết công thức](#viết-công-thức). `background` là mọi thứ nằm ngoài người, ở đây giống `~person`; `all` là mọi bộ phận tìm được, tức là người; `image` là cả bức ảnh và `none` là không có gì. Sau đó `fill_holes`, `mask_expand` và `mask_blur` tinh chỉnh kết quả, theo thứ tự này.
- **Ảnh không có ai.** Khi không tìm thấy khuôn mặt nào, node kiểm tra xem có thật là có người hay không, bằng hai phép thử. Thứ nhất, lượt nhìn đầu của Sapiens2 phải chắc chắn về ít nhất một nửa những gì nó gán là người (một điểm ảnh được tính là chắc chắn từ độ tin cậy 0.9). Thứ hai, một lượt nhìn vào bức ảnh lật gương phải tìm thấy cùng những người đó: phần người của hai lượt nhìn phải chồng lên nhau ít nhất 0.6 (IoU). Người thật vẫn hiện ra trong ảnh lật gương; thứ model tự bịa ra trên một bức tường thì không. Nếu một phép thử không đạt, node coi như bức ảnh không có ai: `skin` rỗng, `background` là cả bức ảnh, và báo cáo cùng bản xem trước ghi "no person found (none Sapiens2 was sure of, and no face)", kèm số thứ tự các ảnh nếu là một batch. Phép thử thứ hai tốn thêm một lượt nhìn, và chỉ khi không tìm thấy khuôn mặt. Nếu không có các phép thử này, một mặt tiền tòa nhà hay một căn phòng có nhiều kệ sẽ hiện ra những "người" mờ nhạt trong tường và trên kệ, và những chỉnh sửa thường ngày trên ảnh mặt tiền (tăng chút độ phơi sáng, độ tương phản, độ bão hòa hay làm nét) có thể đẩy tỉ lệ ở phép thử đầu lên trên một nửa, đó là lý do phép thử thứ hai tồn tại. Người mà khuôn mặt được tìm thấy thì không bao giờ bị bỏ. Một món quần áo đứng một mình (ảnh sản phẩm bày phẳng, không có người mặc) bị tính là không có ai, nên hãy dùng Garment Segment cho trường hợp đó, và một hình người được cách điệu mạnh hoặc bị nhòe, không có khuôn mặt, cũng có thể bị bỏ.
- **Các phép thử "không có ai" được đo thế nào.** Trên 49 ảnh không thấy mặt, 32 ảnh có người (chụp từ phía sau, thân người không có đầu, một ma-nơ-canh) được giữ và 17 ảnh không có ai (một mặt tiền, một quán cà phê, một con voi, ảnh sản phẩm bày phẳng, mặt nạ) bị làm rỗng. Người thật chồng lên ảnh lật gương của họ 0.83 đến 0.99, còn thứ model tự bịa ra trên một mặt tiền chỉ 0.02 đến 0.03. Trên 21 bản chỉnh sửa thường ngày của mỗi ảnh mặt tiền, quán cà phê và con voi, bản đồ luôn rỗng. Trên 21 bản chỉnh sửa của mỗi ảnh trong bốn ảnh có người nhưng không thấy mặt (một ma-nơ-canh, một ảnh sản phẩm không thấy đầu, một vũ công, một sản phẩm đang mặc trên người), bản đồ luôn được giữ.
- **Báo cáo.** Mỗi từ một dòng, rồi các ghi chú nêu ở trên. Với một batch, tỉ lệ tính cho tất cả các ảnh gộp lại, còn bản xem trước dưới mỗi ảnh cho thấy tỉ lệ của riêng ảnh đó.
- **Xem kết quả.** Khi `show_preview` bật, node tự vẽ bản xem trước: không cần node nào khác. Nó vẽ viền các bộ phận mà mask giữ lại: với `skin - hands`, nó vẽ viền các phần da chứ không vẽ bàn tay. Muốn xem các bộ phận ở kích thước đầy đủ của bức ảnh, hãy nối `segs` và đúng bức ảnh đó vào [Segs Preview](#segs-preview-nh).
- **Tự chạy.** Body Segment là một node đầu ra (output node; chấm trên thẻ của nó màu cam): ComfyUI chạy nó kể cả khi không có gì nối vào các đầu ra của nó. Vì vậy một node bạn không còn cần vẫn tiếp tục chạy Sapiens2 mỗi khi đầu vào của nó thay đổi, và tắt `show_preview` không làm nó ngừng chạy. Hãy vô hiệu hóa nó (Mute) hoặc xóa nó.
- **Kết hợp với các node khác.** `segs` chứa cùng các bộ phận dù bạn gõ gì trong `regions`. Hãy nối nó vào [Segs Compose](#segs-compose-nh), cùng với `segs` của SAM3 Segment, Garment Segment hay Portrait Segment chạy trên cùng bức ảnh: ví dụ Body Segment vào `segs`, Portrait Segment vào `segs_2` và `face - eyes - eyebrows` làm biểu thức.

**Cỡ model, bộ nhớ và tốc độ**

| `model_size` | Dung lượng tải | Ghi chú |
|---|---|---|
| `0.4b` (mặc định) | 1.6 GB | Nhanh. Khoảng 1.5 đến 2 GB VRAM ở mức cao nhất. |
| `0.8b` | 3.3 GB | Model lớn hơn: chậm hơn và cần nhiều VRAM hơn. |
| `1b` | 5.9 GB | Lớn nhất: còn chậm hơn nữa và cần nhiều VRAM hơn. |

Trọng số được tải một lần từ Hugging Face (`facebook/sapiens2-seg-<size>`, cố định vào một phiên bản xác định), ngay lần đầu bạn chọn một cỡ, và lưu vào `ComfyUI/models/NH-Nodes/sapiens2-seg-<size>/`, tức thư mục model `nh_nodes`. Chúng ở lại bộ nhớ hệ thống, mỗi lần một cỡ, và chỉ chuyển sang GPU trong lúc node chạy. Node cần PyTorch 2.4 trở lên và dừng lại với một thông báo khi bản của bạn cũ hơn.

**Đo trên một máy** (NVIDIA RTX 5090) với `0.4b`: khoảng 1 giây trên ảnh 1 đến 2K px và 4 đến 7 giây trên ảnh 8K. Chỉ `0.4b` được đo. Trên CPU node vẫn chạy được nhưng chậm: khoảng 8 giây cho mỗi vùng cắt với 8 luồng. Máy khác sẽ cho số khác.

<p align="center"><img src="../assets/examples/body-segment.jpg" alt="Mask của Body Segment trên một ảnh mẫu: da, tóc và môi, quần áo, và phần nền" width="720"><br><em>Body Segment trên một ảnh mẫu, đúng như node hiển thị. Hàng trên: "skin", báo cáo nằm bên dưới bức ảnh. Hàng dưới: "tóc, môi" (từ tiếng Việt), "clothes" và "background"; phần sáng là đầu ra mask của mỗi công thức.</em></p>

**Các công thức để bắt đầu**

| `regions` | Bạn nhận được |
|---|---|
| `skin` | Da của khuôn mặt và cổ, thân, cánh tay, bàn tay, chân và bàn chân. |
| `hair, face` | Tóc, và khuôn mặt cùng cổ. |
| `tóc, môi` | Tóc và môi (từ tiếng Việt). |
| `clothes - shoes` | Thứ người trong ảnh đang mặc, không gồm giày. |
| `skin - hands` | Da nhưng bỏ bàn tay. |
| `skin + lips` | Da kèm cả môi. |
| `background` | Mọi thứ nằm ngoài người. |

**Ví dụ**

```text
Load Image → Body Segment (NH, regions: skin) → mask → Mask Morphology (NH, thu hẹp) → retouch
```

Hãy thu hẹp mask vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh) để phần chỉnh sửa nằm gọn trong da, tránh xa mép tóc và quần áo.

> **Lưu ý:** Ảnh scan cũ, đã phai màu hoặc ngả nâu (sepia) là một giới hạn đã biết. Sapiens2 có thể coi cả phông nền là quần áo: một tấm ảnh nhóm đã scan cho ra 96% diện tích là "người". Hãy dùng Garment Segment (`fashn`) hoặc SAM3 Segment với `person` cho những ảnh như vậy.

> **Giấy phép:** Sapiens2 theo Sapiens2 License của Meta. Giấy phép này miễn phí bản quyền và không có điều khoản cấm thương mại, nhưng cấm theo dõi (surveillance), xử lý sinh trắc học, nhận dạng hay nhận dạng lại người, deepfake, mạo danh và nội dung gây hiểu lầm, suy đoán dữ liệu nhạy cảm về con người khi không có quyền và sự đồng ý mà luật pháp yêu cầu, cùng một số mục đích khác. Nó yêu cầu bạn tuân thủ luật về quyền riêng tư, dữ liệu sinh trắc học và kiểm soát thương mại, và cho phép Meta kiểm tra việc bạn dùng, chấm dứt giấy phép và thay đổi điều khoản. Mã nguồn của NH-Nodes thuộc giấy phép MIT, có kèm mã model của Meta cùng văn bản giấy phép nhưng không có trọng số, vốn được tải từ các kho Hugging Face của Meta. Hãy đọc [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) trước khi dùng node.

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

**Bộ nhớ.** Trọng số được đọc từ đĩa một lần và giữ trong bộ nhớ hệ thống (khoảng 3.6 GB). Chúng chuyển lên GPU khi SAM3 Segment chạy và trở về sau đó, nên ComfyUI nhận lại VRAM. Trước khi chuyển, node nhờ ComfyUI gỡ bớt các model của chính nó nếu VRAM đang eo hẹp. Hãy bật `keep_model_loaded` để trọng số ở lại trên GPU nếu bạn chạy node liên tục. Khi nó bật, nút **Unload Models** của ComfyUI không giải phóng chúng: hãy tắt `keep_model_loaded` và chạy node một lần, hoặc khởi động lại ComfyUI.

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

`NH_SEGS` là dữ liệu mà SAM3 Segment, Garment Segment, Portrait Segment và Body Segment chuyển cho các node Segs. Với mỗi ảnh trong batch (lô ảnh), nó chứa một danh sách instance. Mỗi instance giữ **nhãn** (ở SAM3 Segment là các từ tiếng Anh mà SAM 3 được yêu cầu tìm: `áo đỏ` được lưu thành `red top`; ở ba node kia là tên bộ phận), **điểm tin cậy** (ở ba node kia luôn là 1.0), một **khung** ôm sát và mask **chỉ trong phạm vi khung đó**. Mask kích thước đầy đủ chỉ được dựng khi một node xuất `MASK`, nên hàng chục instance trên ảnh 8K vẫn nhẹ.

- **Đối tượng và bộ phận.** SAM3 Segment trả về mỗi vật nó tìm thấy một instance: ba người là ba instance. Garment Segment, Portrait Segment và Body Segment trả về mỗi bộ phận một instance cho cả bức ảnh: có ba người thì `hair` là tóc của cả ba người gộp lại, trong một instance. Muốn chọn riêng một người trong nhiều người, hãy dùng SAM3 Segment.
- **Nhãn của ba node phân tích người** được liệt kê trong bảng đầu ra của từng node: [Garment Segment](#garment-segment-nh), [Portrait Segment](#portrait-segment-nh) và [Body Segment](#body-segment-nh). Chúng không dùng hết cùng một từ: `mouth` là bên trong miệng ở Portrait Segment, còn Body Segment có `lips`, `teeth` và `tongue`. Các node Segs cũng đọc những từ riêng của node đã tạo ra segs (`skin` trên segs của Body Segment): xem [Segs Compose](#segs-compose-nh). Khi một nhãn không tìm thấy gì, hãy xem các nhãn trong bản xem trước hoặc trong [Segs To Masks](#segs-to-masks-nh).
- Các instance của SAM3 Segment xếp theo thứ tự các nhãn xuất hiện trong concepts của bạn, và theo điểm trong cùng một nhãn. Các instance của ba node kia xếp theo thứ tự danh sách bộ phận của chúng.
- Các node Segs xử lý từng ảnh trong batch riêng biệt.
- Các instance của một ảnh còn giữ một bản in nhỏ màu xám của bức ảnh mà chúng được tìm ra từ đó. [Segs Compose](#segs-compose-nh) dùng nó để từ chối các instance của một bức ảnh khác.
- Các node gốc của ComfyUI không đọc được `NH_SEGS`. Để lấy mask ra, hãy dùng các đầu ra `mask`, [Segs Compose](#segs-compose-nh) hoặc [Segs To Masks](#segs-to-masks-nh). Để dùng detailer của Impact Pack, hãy chuyển đổi bằng [Segs To Impact SEGS](#segs-to-impact-segs-nh).

---

## Viết công thức

`concepts` của SAM3 Segment, `expression` của Segs Compose và `regions` của Body Segment đọc cùng một ngôn ngữ nhỏ. Bạn gọi tên những thứ mình muốn, ghép chúng bằng các toán tử, và kết quả là một mask cho mỗi ảnh. Trong SAM3 Segment, các từ bạn viết cũng là thứ SAM 3 đi tìm. Trong Segs Compose, chúng chọn các instance đã được tìm sẵn. Trong Body Segment, chúng chọn các bộ phận cơ thể mà model của nó gán nhãn.

| Viết | Nghĩa |
|---|---|
| `person` | Một nhãn: các instance mang tên đó. Không phân biệt chữ hoa/thường, và khi so khớp instance thì số ít và số nhiều là một (`arm` tìm `arms`; `glasses` và `glass` vẫn là hai nhãn). |
| `red car`, `women's dress` | Nhiều từ liền nhau là một nhãn, không cần ngoặc kép. Dấu gạch nối nằm giữa hai chữ cái hoặc chữ số thuộc về nhãn (`t-shirt`, `x-ray`), dấu nháy đơn nằm giữa một từ cũng vậy. |
| `person, hair` | Một danh sách: hợp của các mục trong đó. Xuống dòng có tác dụng như dấu phẩy. Danh sách được đọc sau mọi toán tử khác, nên `person - hair, dress` là `person - hair` cộng với `dress`. Mục trống bị bỏ qua. |
| `a + b`, `a \| b` | Hợp: mọi thứ thuộc a hoặc thuộc b. |
| `a - b` | a bỏ đi phần của b. Hãy đặt dấu cách hai bên dấu trừ: `person-hair` được đọc là một nhãn duy nhất. |
| `a & b` | Giao: chỉ nơi cả a và b cùng có. |
| `~a`, `!a` | Mọi thứ trừ a: cả bức ảnh bỏ đi a. |
| `( … )` | Nhóm. |
| `background` | Mọi thứ nằm ngoài mọi instance đã tìm được. Giống `~all`. |
| `all` | Mọi instance đã tìm được. |
| `image` | Cả bức ảnh. |
| `none` | Một mask rỗng. |
| `# …` | Ghi chú, đến hết dòng. Nó bắt đầu ở nơi một từ có thể bắt đầu; nằm giữa một từ, như `car#2`, thì `#` thuộc về từ đó. |
| `"salt & pepper"` | Ngoặc kép giữ chung một nhãn có chứa ký tự toán tử. Một từ `all`, `image`, `none` hay `background` đặt trong ngoặc kép là nhãn thông thường. |

`~` được tính trước, rồi `&`, rồi `+`, `|` và `-` cùng cấp, từ trái sang phải, cuối cùng là danh sách: `a - b + c` nghĩa là `(a - b) + c`. Khi còn phân vân, hãy thêm dấu ngoặc.

**Xuống dòng.** Xuống dòng có tác dụng liệt kê, như dấu phẩy. Nó nối liền khi nằm cạnh một toán tử (dòng trước kết thúc bằng một toán tử, trừ `)`, hoặc dòng sau bắt đầu bằng `+`, `|`, `-`, `&` hay `)`), nên một công thức dài có thể được trình bày trên nhiều dòng:

```text
(wall + ceiling)
  - window
```

**Từ tiếng Việt.** SAM 3 chỉ đọc tiếng Anh, nên các node dịch những từ tiếng Việt thông dụng bạn gõ: `tóc` là `hair`, `váy` là `dress`, `cửa sổ` là `window`. Danh sách gồm con người, cơ thể, quần áo, phụ kiện, thiên nhiên, phòng và công trình, phương tiện, đồ ăn và động vật. Danh sách đầy đủ nằm trong `core/vi_words.py`.

- Một thứ đi kèm màu sắc hoặc kích cỡ được đảo sang thứ tự của tiếng Anh: `áo đỏ` là `red top`, `tóc dài` là `long hair`, `ghế sofa lớn` là `large sofa`. `màu` có thể nằm giữa chúng: `áo màu đỏ`. `tóc vàng` là `blonde hair`.
- `xanh` đứng sau một thứ có thể là xanh dương hoặc xanh lá nên không được đoán. Hãy viết `xanh dương` (blue) hoặc `xanh lá` (green), như `áo xanh dương`, hoặc viết từ tiếng Anh.
- `các`, `những`, `mấy`, `cái`, `chiếc` và `con` đứng trước một thứ bị bỏ qua khi cụm từ không có trong danh sách đúng như đã viết: `những chiếc xe` là `vehicle`.
- `nền`, `phông`, `phông nền` và `hậu cảnh` là từ khóa `background`, cả khi có kèm màu (`nền trắng`). `tất cả` là `all`, còn `toàn ảnh` và `cả ảnh` là `image`.
- Hãy viết tiếng Việt có dấu. Văn bản không có dấu tiếng Việt được coi là tiếng Anh (một vài từ ngắn như `xe` có trong danh sách đúng như đã viết).
- Một cụm tiếng Việt không có trong danh sách sẽ được tìm đúng như đã viết, và SAM 3 thường không tìm thấy gì. Khi đó báo cáo ghi là không hiểu. Nhãn của các instance luôn là từ tiếng Anh.
- Ngoặc kép không chặn việc dịch. `nền` trong ngoặc kép là nhãn `background`, không phải từ khóa.

**Hai node khác nhau ở đâu**

| | SAM3 Segment, `concepts` | Segs Compose, `expression` |
|---|---|---|
| `or`, `and`, `not` | Là từ thông thường, nên một cụm như `black and white dog` đến với SAM 3 đúng như đã viết. | Là toán tử: `a or b`, `a and b`, `not a`. |
| Ký tự đại diện `*` và `?` | Bị từ chối: SAM 3 cần từ thông thường. | Khớp với nhãn (`*shirt`). |
| Nhãn làm gì | Là thứ SAM 3 được yêu cầu tìm. | Chọn các instance có nhãn đó. Nhãn mà không instance nào có cho một phần rỗng. |
| Ô trống | Node dừng lại với thông báo yêu cầu viết ít nhất một concept. Ô chỉ có ghi chú, hoặc chỉ có `all`, `image` hay `none`, cũng được tính là trống. | Node dừng lại với thông báo dùng `all` để lấy mọi instance. |

**`regions` của Body Segment** theo các quy tắc của SAM3 Segment, với ba điểm khác. `or`, `and` và `not` không phải toán tử ở đó (chúng bị từ chối kèm lời khuyên: hãy viết `+`, `-` hoặc `~`). Mỗi từ phải gọi tên một vùng cơ thể mà model của nó có, như `skin`, `hair` hay `clothes`: từ khác, và ký tự đại diện, bị từ chối trước khi nạp model. Và các từ tiếng Việt đi qua cùng danh sách đó, nên ở đó `tay` và `chân` nghĩa là cả chi. Xem [Body Segment](#body-segment-nh).

---

## SAM3 Segment (NH)

<p align="center"><img src="../assets/nodes/NH_SAM3Segment.svg" alt="SAM3 Segment (NH) node" width="360"></p>

Gõ thứ bạn muốn cắt ra và nhận mask của nó: một từ (`person`), một danh sách (`hair, face`) hoặc một công thức (`person - bag`, `background`). SAM 3 tìm mọi instance của những thứ bạn gọi tên. Node trả về mask của công thức, các instance dưới dạng `NH_SEGS`, một con số đếm và một báo cáo cho biết đã tìm thấy và chưa tìm thấy gì, đồng thời hiển thị những gì nó tìm được ngay trên node.

**Dùng khi** bạn muốn có mask của "mọi người", "mọi cửa sổ", "bầu trời", "phần nền" hay "người nhưng không có túi" trong một bức ảnh, kể cả ảnh 4K đến 8K, mà không phải vẽ gì. Với việc chọn tinh hơn (người thứ hai từ trái sang, vật lớn nhất), hoặc để tạo nhiều mask từ một lần chạy, hãy nối `segs` vào các [node Segs](#segs-select-nh).

Các đầu vào COMBO không nêu mặc định sẽ bắt đầu ở lựa chọn đầu tiên.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Bức ảnh. Node dùng màu (RGB), kênh alpha nếu có sẽ bị bỏ. Mọi ảnh trong batch được xử lý lần lượt. |
| `concepts` | STRING | `person` | Thứ cần tìm và mask cần trả về, viết dưới dạng công thức (xem [Viết công thức](#viết-công-thức)): các từ tiếng Anh ngắn (`person`, `red car`, `window`), dấu phẩy hoặc xuống dòng để liệt kê nhiều thứ, và `+`, `-`, `&`, `~`, dấu ngoặc cùng `background` để kết hợp chúng (`person - bag`). Các từ tiếng Việt thông dụng cũng dùng được (`tóc`, `áo đỏ`, `nền`). Hãy viết ít nhất một từ. |
| `threshold` | FLOAT | `0.5` | Khoảng 0.05 đến 0.95. Độ tin cậy tối thiểu. Thấp hơn thì tìm được nhiều instance hơn và cũng nhiều nhầm lẫn hơn; cao hơn chỉ giữ những cái chắc chắn. Điểm đúng bằng ngưỡng thì không được giữ. Khi một concept không tìm thấy gì, báo cáo cho biết ứng viên tốt nhất đã đến gần mức nào. |
| `detail_level` | COMBO | `balanced` | `balanced` hoặc `fast`. `balanced` phân đoạn lại mỗi instance lần thứ hai trên ảnh cận cảnh cắt từ ảnh độ phân giải đầy đủ, để vật nhỏ và vừa có cạnh sắc nét. `fast` giữ lượt đầu trên cả bức ảnh. Xem [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động). |
| `max_per_concept` | INT | `0` | Khoảng 0 đến 200. Giữ tối đa bấy nhiêu instance cho mỗi concept, điểm cao nhất trước. `0` giữ tất cả. |
| `checkpoint` | COMBO | — | Một tệp `.pt` hoặc `.pth` của SAM 3 trong `models/sam3` hoặc thư mục khác đã đăng ký với tên `sam3`. Các pack SAM 3 khác dùng chung thư mục này. Khi thư mục chưa có tệp nào, danh sách hiện `sam3.pt` và node cho bạn biết lấy nó ở đâu khi bạn chạy. |
| `keep_model_loaded` | BOOLEAN | `false` | Tắt: sau lần chạy, trọng số trở về bộ nhớ hệ thống và ComfyUI nhận lại VRAM. Bật: chúng ở lại trên GPU (khoảng 3.6 GB), giúp lần chạy sau nhanh hơn, nhưng ComfyUI không dùng được phần VRAM đó trong lúc ấy, và nút **Unload Models** của nó không giải phóng được (hãy tắt tùy chọn này rồi chạy node một lần, hoặc khởi động lại ComfyUI). |
| `show_preview` | BOOLEAN | `true` | Tùy chọn (optional). Hiển thị kết quả ngay trên node: mask sáng và phần còn lại bị làm tối, các instance có viền và gắn nhãn, và các dòng báo cáo bên dưới. Tối đa 1536 px ở cạnh dài, và chỉ 16 ảnh đầu của một batch. Tắt: không vẽ gì, và node vẫn chạy (xem bên dưới). |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `segs` | NH_SEGS | Các instance của từng ảnh, mỗi instance có nhãn, điểm, khung và mask. Nhãn là các từ tiếng Anh mà SAM 3 được yêu cầu tìm. Khi công thức dùng `background`, chúng bao gồm cả các instance `person` đã được tìm cho nó. Hãy nối nó vào các node Segs. |
| `mask` | MASK | Mask của công thức, một mask cho mỗi ảnh, có kích thước bằng ảnh. Một danh sách từ đơn thuần cho hợp của mọi instance. |
| `count` | INT | Số instance tìm được, tính trên cả batch, gồm cả các instance `person` mà `background` đã tìm. |
| `report` | STRING | Những gì yêu cầu đã tìm được, mỗi từ một dòng: số lượng và điểm cao nhất, bản dịch, các trường hợp suýt đạt, các từ không hiểu, và ý nghĩa của `background`. Xem bên dưới. |

<p align="center"><img src="../assets/examples/sam3-formula.jpg" alt="Thứ SAM3 Segment hiển thị trên node với bốn công thức trên một ảnh mẫu: background, person - face, hair và face, và mọi thứ trừ người và các mũ bảo hiểm" width="720"><br><em>SAM3 Segment trên một ảnh mẫu, đúng như node hiển thị. Hàng trên: "background" làm sáng mọi thứ nằm ngoài người, và báo cáo nằm bên dưới bức ảnh. Hàng dưới: ba công thức khác; phần sáng là đầu ra mask của mỗi công thức.</em></p>

**Báo cáo.** Dòng đầu của `report` cho tổng số. Mỗi dòng sau đó (chúng cũng được viết bên dưới bản xem trước) bắt đầu bằng thứ bạn đã viết:

| Dòng | Nghĩa |
|---|---|
| `person: 2 found (best 0.91)` | Số instance tìm được và điểm cao nhất. |
| `áo đỏ -> red top: 1 found (best 0.64)` | Cụm tiếng Việt đã được dịch thành gì. |
| `hair: none at threshold 0.50; the best candidate scored 0.31, lower the threshold to keep it` | Không có gì vượt `threshold`, nhưng có một ứng viên suýt đạt. Hãy hạ `threshold` xuống dưới điểm của nó để giữ lại. |
| `bag: none found; try another English word` | Không ứng viên nào đạt nổi 0.1. |
| `áo xanh: not understood; write it in English (SAM 3 reads English only)` | Một cụm tiếng Việt không có trong danh sách và không tìm thấy gì. |
| `background = everything outside person, red top` | Ý nghĩa của `background`: mọi thứ nằm ngoài các instance của những nhãn này. |
| `person (searched for background): 2 found` | Lần tìm `person` mà node thêm vào vì công thức dùng `background` mà không nêu `person`. |

**Cách hoạt động**

- **Nhãn.** Các từ tiếng Anh mà SAM 3 được yêu cầu tìm là nhãn của các instance: `áo đỏ` được tìm là `red top`, và các instance của nó mang nhãn `red top`. Các node Segs bỏ qua chữ hoa/thường khi so khớp nhãn, nên cách viết hoa không thành vấn đề.
- **Mask.** `mask` là kết quả của công thức. Với `person - bag`, SAM 3 tìm `person` và `bag`, `mask` là người trừ đi các túi, và `segs` vẫn chứa cả hai loại instance.
- **`background`.** Là mọi thứ nằm ngoài mọi instance đã tìm được, nên node cần biết chủ thể nằm ở đâu. Khi công thức dùng nó mà không nêu `person`, SAM 3 cũng tìm `person`; các instance đó thuộc `segs` và `count` và có viền trong bản xem trước. Muốn loại cả những thứ khác ra khỏi phần nền, hãy nêu tên chúng: `~(person + wine glass + chair)`. Danh sách là phép hợp, nên `person, background` là cả bức ảnh.
- **Thứ tự và bản trùng.** Các instance xếp theo thứ tự các nhãn xuất hiện trong công thức (`person` do `background` thêm vào đứng cuối), điểm cao nhất trước trong mỗi nhãn. Hai instance của cùng một nhãn mà mask gần như trùng hoàn toàn (từ 90% trở lên) được tính là một, giữ lại cái điểm cao hơn. Sau đó `max_per_concept` giữ những cái tốt nhất trong phần còn lại. Các instance có điểm đúng bằng nhau giữ thứ tự mà SAM 3 đưa ra. Một concept viết cả số ít lẫn số nhiều (`arm, arms`) chỉ được tìm một lần, và báo cáo cũng chỉ liệt kê nó một lần; `glass` và `glasses` vẫn là hai concept.
- **Xem kết quả.** Khi `show_preview` bật, node tự vẽ bản xem trước: không cần node nào khác. Muốn xem ảnh ở kích thước đầy đủ (bản xem trước tối đa 1536 px), hoặc xem Segs Select đang làm việc với gì, hãy nối `segs` và đúng bức ảnh đó vào [Segs Preview](#segs-preview-nh).
- **Tự chạy.** SAM3 Segment là một node đầu ra (output node; chấm trên thẻ của nó màu cam): ComfyUI chạy nó kể cả khi không có gì nối vào các đầu ra của nó. Vì vậy một node bạn không còn cần vẫn tiếp tục chạy SAM 3 mỗi khi đầu vào của nó thay đổi, và tắt `show_preview` không làm nó ngừng chạy. Hãy vô hiệu hóa nó (Mute) hoặc xóa nó. Chừng nào đầu vào giữ nguyên, ComfyUI dùng lại kết quả lần trước.
- **Không tìm thấy gì** không phải lỗi: bạn nhận được không instance nào và `count` bằng 0. Mask rỗng với danh sách từ đơn thuần, và là cả bức ảnh với một công thức như `background` hay `~person`, vì không có gì được tìm thấy để loại ra.
- **Batch.** Mỗi ảnh được xử lý lần lượt và có danh sách instance riêng. Thanh tiến trình đếm theo concept, và bạn có thể hủy lần chạy giữa các concept và giữa các ảnh cận cảnh.
- **Vật bị khung hình cắt.** SAM 3 làm mờ dần mask của nó trong một dải dọc biên của hình vuông nó làm việc, rộng khoảng 0.3% cạnh của bức ảnh. Mask của một vật tràn ra khỏi ảnh vẫn chạm tới mép ảnh: dải đó được nâng lên ở nơi vật chạm tới nó từ bên trong. Cái giá là một vật kết thúc trong dải đó, ngay phía trên mép ảnh, bị vẽ kéo xuống tận mép.
- **Ảnh rất rộng hoặc rất cao.** Trước hết SAM 3 nhìn cả bức ảnh bị nén vào một hình vuông. Với ảnh rộng hoặc cao hơn 4:1 (ảnh toàn cảnh), nó có thể bỏ sót vật nhỏ, và console cho biết điều đó. Hãy chia ảnh như vậy thành các ô bằng [Image Tile (NH)](./image-compose.md#image-tile-nh) rồi chạy từng ô.
- **Lỗi cài đặt** làm node dừng lại với thông báo nói rõ cần làm gì: thiếu gói, thiếu checkpoint, không có GPU CUDA. Xem [Khi có sự cố với SAM 3](#khi-có-sự-cố-với-sam-3).

> **Mẹo:** Hãy viết danh từ tiếng Anh đơn giản: `window`, `wine glass`, `red car`. Nếu một concept không tìm thấy gì, hãy thử từ khác (một tòa nhà có hình dạng lạ có thể đáp ứng `structure` hoặc `monument` thay vì `building`) hoặc hạ `threshold`. Báo cáo cho biết đó là trường hợp nào: không tìm thấy gì, một ứng viên suýt đạt, hay một từ không được hiểu.

> **Mẹo:** Hãy yêu cầu cả tổng thể lẫn các phần của nó trong cùng một công thức: `person - hair`. Node tìm cả hai và `mask` là phần chênh lệch. `segs` vẫn chứa cả hai, nên [Segs Compose](#segs-compose-nh) có thể tạo các mask khác từ cùng lần chạy đó.

> **Giấy phép:** SAM 3 thuộc SAM License của Meta. Xem [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3).

---

## Segs Select (NH)

<p align="center"><img src="../assets/nodes/NH_SegsSelect.svg" alt="Segs Select (NH) node" width="360"></p>

Giữ lại những instance bạn muốn, theo nhãn, điểm, kích thước và vị trí, theo thứ tự bạn chọn.

**Dùng khi** bạn chỉ cần một phần những gì SAM3 Segment tìm được: vật lớn nhất, người thứ hai từ trái sang, chỉ những kết quả chắc chắn, hay chỉ những ô cửa lớn.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment, Garment Segment, Portrait Segment, Body Segment hoặc một node Segs khác. |
| `labels` | STRING | rỗng | Các nhãn cần giữ, cách nhau bằng dấu phẩy. `*` thay cho một chuỗi ký tự bất kỳ và `?` thay cho một ký tự (`*shirt`). Không phân biệt chữ hoa/thường, số ít và số nhiều là một (`arm` tìm `arms`), và `_` được coi là dấu cách (`red_car` tìm `red car`). Nhãn có dấu cách không cần ngoặc kép ở đây: `wine glass, red car`. Các từ tiếng Việt thông dụng được dịch (`tóc` tìm `hair`); các từ công thức `nền` và `tất cả` không phải nhãn ở đây. Các từ riêng của node đã tạo ra segs cũng dùng được: `skin` trên segs của Body Segment giữ các bộ phận face, torso, arms, hands, legs và feet của nó (xem [Segs Compose](#segs-compose-nh)). Để trống thì giữ mọi nhãn. |
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

Tạo một mask từ nhãn của các instance, bằng chính công thức nhỏ của `concepts` trong SAM3 Segment: hợp, trừ, giao và "mọi thứ trừ...". Các instance có thể đến từ tối đa ba node cùng lúc.

**Dùng khi** bạn đã có sẵn các instance và muốn có "phần nền", "người nhưng không có tóc", "bức tường nhưng không có cửa sổ", hoặc một tổ hợp khác của những gì SAM3 Segment tìm được: sau [Segs Select](#segs-select-nh), hoặc để tạo nhiều mask từ một lần chạy SAM 3. Cũng dùng nó để trộn những gì nhiều node tìm được trên cùng một bức ảnh: cái túi từ SAM 3, chiếc đầm từ Garment Segment, đôi môi từ Body Segment. Với một mask duy nhất chỉ từ SAM 3, bạn cũng có thể viết đúng công thức đó thẳng vào `concepts` của SAM3 Segment.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment, Garment Segment, Portrait Segment, Body Segment hoặc Segs Select. |
| `expression` | STRING | `all` | Công thức. Xem [Viết công thức](#viết-công-thức). Ô này có nhiều dòng: xuống dòng có tác dụng liệt kê, như dấu phẩy, và nối liền khi nằm cạnh một toán tử. |
| `segs_2` | NH_SEGS | — | Tùy chọn (optional). Thêm các instance của cùng bức ảnh, từ một trong các node trên. Chúng được trộn vào công thức cùng với `segs`. |
| `segs_3` | NH_SEGS | — | Tùy chọn (optional). Và thêm nữa. |
| `check_same_photo` | BOOLEAN | `true` | Tùy chọn (optional). Từ chối các segs trông như từ những bức ảnh khác nhau có cùng kích thước. Hãy tắt nó khi cần trộn segs của các bản chỉnh sửa từ một bức ảnh mà khác nhau nhiều, như đổi phông nền hay đổi ánh sáng. Xem "Chỉ cùng một bức ảnh" bên dưới. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Kết quả của công thức, một mask cho mỗi ảnh, có kích thước bằng ảnh. |

**Ở đây có gì khác.** Ngôn ngữ chính là ngôn ngữ trong [Viết công thức](#viết-công-thức), với những khác biệt sau so với SAM3 Segment:

- `or`, `and` và `not` là toán tử: `a or b` là `a | b`, `a and b` là `a & b`, `not a` là `~a`.
- Một nhãn chọn các instance mang nhãn đó. Không phân biệt chữ hoa/thường, số ít và số nhiều là một (`arm` tìm `arms`, `foot` tìm `feet`, `shoe` tìm `shoes`, `scarf` tìm `scarves`; `glasses` và `glass` thì được giữ riêng), `_` được coi là dấu cách (`red_car` tìm `red car`), và `*`, `?` là ký tự đại diện (wildcard), được so khớp đúng như đã viết (`*shirt`).
- Đặt nhãn có chứa ký tự toán tử trong ngoặc kép, `"` hoặc `'` (`"salt & pepper"`), và dùng ngoặc kép đôi khi nhãn còn có dấu nháy đơn. Nhiều từ liền nhau không cần ngoặc kép.
- Nhãn tiếng Việt được dịch (`tóc` tìm `hair`), cả khi nằm trong ngoặc kép. `nền` là từ khóa `background`.

| Biểu thức | Bạn nhận được |
|---|---|
| `person - hair` | Người nhưng bỏ phần tóc (hãy yêu cầu SAM3 Segment `person, hair`). |
| `~person` | Mọi thứ trừ người. |
| `background` | Mọi thứ nằm ngoài mọi instance trong segs. Khác với `~person`, nó cũng loại cả cửa sổ, ghế hay bất cứ thứ gì khác đã tìm được. |
| `(wall \| ceiling) - window` | Tường và trần, khoét bỏ các ô cửa sổ. |
| `wine glass, person` | Các ly rượu và người cùng lúc (một danh sách, giống `wine glass \| person`). |
| `all - person` | Mọi thứ đã tìm được trừ người. Khác với `~person`, nó không gồm phần nền mà không instance nào phủ tới. |
| `dress - bag - lips` | Với ba node trên cùng một bức ảnh: SAM3 Segment (`bag`) vào `segs`, Garment Segment vào `segs_2` và Body Segment vào `segs_3`. Chiếc đầm nhưng bỏ mọi cái túi và đôi môi. |
| `face - eyes - eyebrows` | Body Segment vào `segs` và Portrait Segment vào `segs_2`. `face` lấy khuôn mặt của cả hai node (xem bảng kế tiếp), còn mắt và lông mày của Portrait Segment bị bỏ ra. |
| `skin - bag` | SAM3 Segment (`bag`) vào `segs` và Body Segment vào `segs_2`. Da nhưng bỏ cái túi: `skin` là một từ riêng của Body Segment (xem bảng kế tiếp). |
| `skin - hands` | Chỉ cần segs của Body Segment, với cùng các từ đó. Da nhưng bỏ bàn tay. |

**Từ riêng của một node.** Một nhãn chọn các instance mang nhãn đó. Mỗi node được đọc riêng: ở node nào không có instance mang từ đó làm nhãn, Segs Compose đọc từ đó như một từ riêng của node ấy và lấy các bộ phận mà từ đó đại diện. Instance của một node mang từ đó không che mất từ của node khác: khi segs của Portrait Segment và của Body Segment cùng có mặt, `skin` là instance skin của Portrait Segment cộng với các bộ phận da của Body Segment. [Segs Select](#segs-select-nh) đọc cùng những từ này trong `labels`. Chỉ các từ nhóm mới đại diện cho nhiều bộ phận:

| Với segs của | Từ nhóm | Chúng lấy gì |
|---|---|---|
| Body Segment | `skin` (`body skin`), `head`, `mouth`, `clothes` (`clothing`, `outfit`), `person` (`body`), các từ tiếng Việt tương ứng (`da`, `đầu`, `miệng`, `quần áo`, `người`), và các chi nguyên vẹn `tay`, `đôi tay`, `hai tay`, `chân`, `đôi chân`, `hai chân` | `skin`: các bộ phận face, torso, arms, hands, legs và feet. `head`: face và hair. `mouth`: lips, teeth và tongue. `clothes`: top, bottom, shoes, socks và accessories. `person`: mọi bộ phận. `tay`, `đôi tay`, `hai tay`: arms và hands. `chân`, `đôi chân`, `hai chân`: legs và feet. |
| Garment Segment | Các nhóm của `part`: `upper_garment`, `lower_garment`, `footwear`, `headwear`, `all_garments` | Với `fashn`: `upper_garment` là top, dress và scarf, `lower_garment` là skirt, pants và belt, `footwear` là feet, `headwear` là hat, `all_garments` là tất cả những thứ đó. Với `fashionpedia`: các nhóm riêng của nó (`footwear` là shoe và sock, `headwear` là hat và headband…). |
| Portrait Segment | `face`, `face skin`, `mouth interior` | `face`: skin, nose, glasses, eyes, eyebrows, mouth và lips. `face skin`: bộ phận skin. `mouth interior`: bộ phận mouth, tức bên trong miệng. |

- Một từ trùng tên với một bộ phận của chính node đó thì không bao giờ đại diện cho các bộ phận khác: `belt` của Fashionpedia chỉ là chiếc thắt lưng.
- Các từ gần nghĩa trong `regions` của Body Segment (`dress`, `pants`, `skirt`, `jeans`, `shirt`, `neck`…) không được đọc ở đây: chỉ với segs của Body Segment, `dress` hay `váy` và `pants` hay `quần` không tìm thấy gì. Hãy viết tên bộ phận (`top`, `bottom`), hoặc trộn thêm segs của Garment Segment, nơi dress, skirt và pants được gắn nhãn thật.
- Một từ chỉ phủ một phần của một bộ phận (`left hand`, `upper arm`, `thigh`) thì không tìm thấy gì ở đây, các dạng một bên `tay trái`, `tay phải`, `chân trái` và `chân phải` cũng vậy: segs của Body Segment không có instance riêng cho một bên. `regions` của Body Segment có những từ đó. `cánh tay` vẫn là chỉ cánh tay, không kèm bàn tay.
- Segs Select đọc từ tiếng Việt `tay` và `chân` là `arm` và `leg`, tức cánh tay và cái chân không kèm bàn tay và bàn chân.

**Cách hoạt động**

- **Nhãn mà không instance nào có** cho một phần rỗng, và console báo điều đó kèm danh sách các nhãn mà segs đang có (`no instance is labelled 'x'; those parts are empty. The segs hold: hair, face, …`). Vì thế gõ sai `~persn` sẽ cho cả bức ảnh: hãy xem console khi mask trông lạ.
- **Các từ liền nhau là một nhãn.** Khi không instance nào có nhãn đó nhưng mỗi từ lại là nhãn của một instance (`person hair`), node dừng với một lỗi gợi ý `person + hair`. Với dấu gạch nối cũng vậy: khi cả `person` và `hair` đều có, `person-hair` làm node dừng với lỗi cho thấy `person - hair`.
- **Mỗi ảnh** trong batch có mask riêng, dựng từ chính các instance của nó.
- **Trộn nhiều node.** `segs`, `segs_2` và `segs_3` được gộp lại trước khi công thức được đọc: mỗi ảnh nhận các instance của tất cả chúng, và một nhãn chọn các instance của mọi node có nhãn đó. Khi hai node dùng cùng một nhãn (`hair` ở Portrait Segment và ở Body Segment, `glasses` ở ba node), nhãn đó là hợp mask của chúng. Mỗi node cắt bức ảnh bằng model riêng của nó, nên mask từ các node khác nhau không khớp nhau đến từng cạnh.
- **Chỉ cùng một bức ảnh.** Node dừng lại khi các segs đến từ các bức ảnh khác nhau: số ảnh khác nhau, hoặc kích thước khác nhau. Khi `check_same_photo` bật, node còn dừng lại nếu chúng trông như những bức ảnh khác nhau có cùng kích thước. Mỗi khung (frame) của segs giữ một bản in nhỏ màu xám của bức ảnh của nó, và các bản in phải khớp nhau. Ảnh đã chỉnh sửa từ cùng một bức (độ sáng, độ tương phản, màu sắc, làm mờ) vẫn được chấp nhận. Đây là chốt chặn cho việc nối nhầm dây chứ không phải bằng chứng: nó bắt được phần lớn các bức ảnh khác có cùng kích thước, một bản bị dịch hoặc phóng nhẹ thường vẫn lọt qua, và một phông nền mới sau cùng một người có thể bị từ chối. Với những bản chỉnh sửa như vậy, hãy tắt `check_same_photo`. Segs do phiên bản cũ tạo ra không có bản in và được trộn chỉ dựa vào kích thước.
- **Công thức không đọc được** làm node dừng lại với thông báo nêu vị trí của chỗ sai, ví dụ `unexpected '(' at position 8` với `person (adult)`.

<p align="center"><img src="../assets/examples/sam3-segment.jpg" alt="Các instance của SAM3 Segment do Segs Preview vẽ, và ba mask tạo từ chúng bằng Segs Compose và Segs Select trên một ảnh mẫu" width="720"><br><em>Các instance của "person, helmet, face" trên một ảnh mẫu, do Segs Preview vẽ, rồi ba mask tạo từ chúng bằng Segs Compose và Segs Select.</em></p>

---

## Segs To Masks (NH)

<p align="center"><img src="../assets/nodes/NH_SegsToMasks.svg" alt="Segs To Masks (NH) node" width="360"></p>

Đổi các instance thành mask thông thường: một mask cho mỗi instance, mask hợp của từng ảnh, và danh sách nhãn.

**Dùng khi** bạn cần mỗi instance là một mask riêng: để xử lý từng cái một, để lưu lại, hoặc để đọc nhãn của chúng.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment, Garment Segment, Portrait Segment, Body Segment hoặc Segs Select. |

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
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment, Garment Segment, Portrait Segment, Body Segment hoặc Segs Select. |
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

**Dùng khi** bạn muốn xem các instance ở kích thước đầy đủ của bức ảnh, hoặc xem Segs Select đã chọn gì. SAM3 Segment đã hiển thị một bản xem trước nhỏ ngay trên node (tối đa 1536 px, kèm báo cáo). Hãy đặt node này sau SAM3 Segment để xem tất cả, hoặc sau Segs Select để chỉ xem phần đã chọn.

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

Trong mỗi workflow dưới đây, hãy kiểm tra những gì SAM3 Segment tìm được trước khi chỉnh sửa bất cứ thứ gì. Bản xem trước trên node cho thấy mọi instance với viền và nhãn, và báo cáo của nó cho biết cái gì không tìm thấy. Nếu thiếu một instance bạn muốn, hãy hạ `threshold` (ví dụ xuống 0.3) hoặc thử từ khác cho concept.

### Nền phía sau người

```text
Load Image → SAM3 Segment (concepts: background)
           → Mask Morphology (NH)
           → chỉnh sửa phần nền
```

`background` là mọi thứ nằm ngoài những người SAM 3 tìm thấy: node tự tìm `person`. Khi người đang cầm hoặc ngồi lên thứ gì, hãy nêu tên cả những thứ đó để chúng cũng nằm ngoài phần nền, ví dụ concepts `~(person + wine glass + chair)`. Dịch cạnh của mask vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh): mở rộng để phần chỉnh sửa phủ lên đường viền cũ, hoặc thu hẹp nếu phần chỉnh sửa lấn vào người. Sau đó dùng mask này để làm mờ, đổi ánh sáng, đổi màu hoặc inpaint phần nền.

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
Load Image → SAM3 Segment (concepts: building - (window + door))
           → đổi màu, đổi phong cách hoặc inpaint bức tường
```

`building` phủ cả mặt tiền cùng cửa sổ và cửa ra vào; công thức khoét chúng ra và chừa lại bức tường. Với một căn phòng, hãy dùng `(wall + ceiling) - window`. Nếu `building` không tìm thấy gì trên một tòa nhà có hình dạng lạ, hãy thử `structure` hoặc `monument` và sửa công thức cho khớp. Trên ảnh 4K đến 8K, hãy giữ `balanced`: mỗi cửa sổ được phân đoạn lại trên một ảnh cận cảnh của các điểm ảnh gốc. Đường viền của một mặt tiền chiếm trọn khung hình chỉ chính xác đến khoảng 1/300 bề ngang của nó (xem [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động)), nên hãy kiểm tra đường mái trong bản xem trước.

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
| Lỗi báo bạn phải viết ít nhất một concept | `concepts` đang trống, chỉ có dòng ghi chú, hoặc chỉ có các từ khóa `all`, `image` hay `none`. Hãy viết một từ như `person`. |
| Lỗi báo một concept có ký tự đại diện (wildcard) | Hãy bỏ `*` hoặc `?`: SAM 3 cần từ thông thường. Ký tự đại diện dùng được trong Segs Select và Segs Compose. |
| Lỗi từ SAM3 Segment có nêu một vị trí, như `unexpected '(' at position 8` | Kiểm tra công thức tại vị trí đó. Dấu ngoặc trong một nhãn được đọc là nhóm. Xem [Viết công thức](#viết-công-thức). |
| `count` bằng 0, hoặc thiếu một instance | Hãy đọc báo cáo: nó cho biết từ đó không tìm thấy gì, một ứng viên suýt đạt (hạ `threshold` xuống điểm của nó), hay từ đó không được hiểu (hãy viết bằng tiếng Anh). Bản xem trước trên node cho thấy những gì đã tìm được. |
| Báo cáo ghi một từ không được hiểu | Đó là một cụm tiếng Việt không có trong danh sách. Hãy viết bằng tiếng Anh. Sau một thứ, `xanh` không rõ nghĩa: hãy viết `xanh dương` (blue) hoặc `xanh lá` (green). |
| Mask là cả bức ảnh, hoặc không như bạn mong đợi | Danh sách là phép hợp: `person, background` là người cộng với mọi thứ nằm ngoài người. Hãy viết riêng `background`, hoặc một công thức như `person - bag`, và nhìn bản xem trước: phần sáng là mask. |
| SAM3 Segment vẫn chạy dù không có gì nối vào nó | Nó là node đầu ra (output node). Hãy vô hiệu hóa (Mute) hoặc xóa nó khi không cần; tắt `show_preview` không làm nó ngừng chạy. |
| Console báo đã giữ mask lượt đầu cho một số instance | Ảnh cận cảnh của chúng không khớp với lượt đầu, nên các instance đó giữ cạnh của lượt đầu. Không có gì sai. |
| Lỗi từ Segs Preview về kích thước hoặc số ảnh | Hãy nối đúng bức ảnh mà segs được tìm ra từ đó. |
| Lỗi từ Segs Compose có nêu một vị trí | Kiểm tra công thức tại vị trí đó. Xem [Viết công thức](#viết-công-thức). |
| Lỗi từ Segs Compose báo không instance nào có nhãn `person hair`, nhưng `person` và `hair` thì có | Các từ liền nhau là một nhãn. Hãy nối chúng lại: `person + hair`. |
| Mask của Segs Compose rỗng, và console báo không instance nào có một nhãn | Kiểm tra chính tả của nhãn, và đọc các nhãn mà console liệt kê sau "The segs hold". Đặt dấu cách quanh dấu trừ: `person - hair`, không phải `person-hair`. Nhãn khác nhau giữa các node: xem bảng đầu ra của node đã tạo ra segs, và các từ riêng của nó ở mục "Từ riêng của một node" trong [Segs Compose](#segs-compose-nh). |
| Lỗi từ Segs Compose báo các segs cần trộn đến từ các bức ảnh khác nhau (different pictures) | `segs`, `segs_2` và `segs_3` phải đến từ cùng một batch ảnh: cùng số ảnh, cùng kích thước. Hãy chạy mọi node trên cùng một Load Image. |
| Lỗi từ Segs Compose báo các segs cần trộn trông như những bức ảnh khác nhau có cùng kích thước (look like different photos of the same size) | Một trong các node đã chạy trên một bức ảnh khác có cùng kích thước. Hãy chạy mọi node trên cùng một bức ảnh. Nếu chúng là các bản chỉnh sửa của một bức ảnh mà khác nhau nhiều (đổi phông nền, đổi ánh sáng), hãy tắt `check_same_photo`. Bản chỉ chỉnh độ sáng, độ tương phản, màu sắc hoặc làm mờ thì được chấp nhận. |
| Lỗi từ Segs To Masks báo các mask instance kích thước đầy đủ cần nhiều bộ nhớ hơn mức còn trống | Có quá nhiều instance so với kích thước bức ảnh. Dùng Segs Select để chỉ giữ những instance bạn cần, hoặc Segs Compose để lấy một mask duy nhất. |
| Console báo SAM 3 nhìn bức ảnh bị nén vào hình vuông và có thể bỏ sót vật nhỏ | Bức ảnh rộng hoặc cao hơn 4:1. Hãy chia nó thành các ô bằng Image Tile (NH) rồi chạy từng ô. |
| ComfyUI hết VRAM sau khi node chạy | Để `keep_model_loaded` tắt, để trọng số rời GPU sau mỗi lần chạy. |
| Nút Unload Models của ComfyUI không giải phóng VRAM của SAM 3 | `keep_model_loaded` đang bật. Hãy tắt nó và chạy node một lần, hoặc khởi động lại ComfyUI. |

---

[← Về trang hướng dẫn](./README.md)
