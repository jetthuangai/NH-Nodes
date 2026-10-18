[← Về trang hướng dẫn](./README.md)

# Vision (phân đoạn ảnh)

Tám node cắt từng phần trên ảnh chụp. Hai node dùng các model SegFormer hiểu về con người: trang phục với Garment Segment, và khuôn mặt, tóc, cổ với Portrait Segment. Mỗi node trả về một mask có cùng kích thước với ảnh đầu vào, một ảnh cắt nền có độ trong suốt và một ảnh xem trước mask. Sáu node còn lại dùng SAM 3, model phân đoạn của Meta: bạn gõ thứ mình muốn vào SAM3 Segment (người, ly rượu, cửa sổ, bầu trời, hoặc một công thức như `person - bag` hay `background`) và nó trả về mask, đồng thời hiển thị ngay trên node. Nó cũng giữ lại mọi instance (từng đối tượng riêng lẻ) tìm được, và năm node Segs giúp chọn, ghép, xem trước và chuyển đổi chúng. Bạn tìm thấy chúng trong **NH-Nodes → Vision**.

**Trên trang này**

- [Nên dùng node nào?](#nên-dùng-node-nào) — các node SegFormer hay SAM 3
- **Các node SegFormer: trang phục và chân dung**
  - [Trước khi bắt đầu](#trước-khi-bắt-đầu) — model, bộ nhận diện khuôn mặt, lần chạy đầu tiên, giấy phép, bộ nhớ
  - [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh) — vì sao khuôn mặt và người nhỏ được xem cận cảnh
  - [Tinh chỉnh mask](#tinh-chỉnh-mask) — các tùy chọn dùng chung của hai node
  - [Garment Segment (NH)](#garment-segment-nh) — mask trang phục thân trên, thân dưới, giày dép hoặc mũ nón
  - [Portrait Segment (NH)](#portrait-segment-nh) — mask khuôn mặt, tóc, cổ và các bộ phận trên mặt
- **Các node SAM 3: bất cứ thứ gì bạn gọi tên được**
  - [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3) — GPU, gói cài đặt, checkpoint, bộ nhớ, giấy phép
  - [Cách hai lượt phân tích hoạt động](#cách-hai-lượt-phân-tích-hoạt-động) — `balanced` làm được gì trên ảnh 4K đến 8K, và giới hạn ở đâu
  - [Các instance: kiểu NH_SEGS](#các-instance-kiểu-nh_segs) — thứ được chuyển qua lại giữa các node
  - [Viết công thức](#viết-công-thức) — những gì `concepts` và `expression` nhận: danh sách, `-`, `&`, `~`, `background`, từ tiếng Việt
  - [SAM3 Segment (NH)](#sam3-segment-nh) — gõ thứ bạn muốn, nhận mask của nó và xem ngay trên node
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
| Trang phục của một người hoặc cả nhóm: áo, quần, giày, mũ | [Garment Segment](#garment-segment-nh) | Một model SegFormer và một bộ nhận diện khuôn mặt nhỏ, đều tự tải ở lần chạy đầu. Chạy được trên CPU hoặc GPU. |
| Khuôn mặt, tóc, cổ và các bộ phận trên mặt của một ảnh chân dung, ảnh nửa người hoặc mọi khuôn mặt trong ảnh nhóm | [Portrait Segment](#portrait-segment-nh) | Giống như trên. |
| Bất cứ thứ gì bạn gọi tên được (người, đồ vật, bầu trời, tường, cửa sổ) dưới dạng mask từ một công thức ngắn (`person - bag`, `background`), hoặc từng instance riêng, kể cả trên ảnh 4K đến 8K | [SAM3 Segment](#sam3-segment-nh) và các node Segs | GPU NVIDIA, gói `sam3` của Meta và một checkpoint mà bạn tự cài và tự tải. Xem [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3). |

Bạn cũng có thể yêu cầu SAM3 Segment tìm `dress` hay `hair`. Các node SegFormer vẫn là lựa chọn dễ hơn cho các bộ phận trang phục và khuôn mặt cố định của chúng: chúng tự thiết lập và cũng chạy được không cần GPU.

## Trước khi bắt đầu

Phần này nói về Garment Segment và Portrait Segment. SAM3 Segment và các node Segs có phần riêng: [Trước khi bắt đầu với SAM 3](#trước-khi-bắt-đầu-với-sam-3).

| | Garment Segment (NH) | Portrait Segment (NH) |
|---|---|---|
| Model | SegFormer huấn luyện cho các lớp trang phục của Fashionpedia | SegFormer tinh chỉnh (fine-tune) trên CelebAMask-HQ cho các bộ phận khuôn mặt |
| Kho Hugging Face | `1038lab/segformer_fashion` | `jonathandinu/face-parsing` |
| Dung lượng tải | khoảng 190 MB | khoảng 340 MB |
| Giấy phép | Người đăng tải không khai báo | Chỉ dành cho nghiên cứu và giáo dục phi thương mại |

**Lần chạy đầu tiên.** Khi chạy lần đầu, mỗi node tải model của nó từ Hugging Face, cùng với bộ nhận diện khuôn mặt mô tả bên dưới. Mỗi lần tải được cố định (pinned) vào một phiên bản xác định của kho, nên thay đổi về sau ở nguồn không thể làm đổi thứ bạn nhận được. Các tệp được lưu vào `ComfyUI/models/NH-Nodes/`, mỗi model một thư mục con. Nếu đã có sẵn một bản đầy đủ thì node dùng luôn bản đó: trong `models/RMBG/segformer_fashion` với model trang phục, trong `models/face_parsing` hoặc `models/RMBG/segformer_face` với model khuôn mặt, hoặc trong bất kỳ thư mục nào bạn thêm bằng khóa `nh_nodes` trong `extra_model_paths.yaml` (xem ví dụ ở mục [Installation](../../README.md#installation)). Nếu quá trình tải bị gián đoạn, hãy chạy lại node: các tệp đã tải xong được giữ lại.

**Bộ nhận diện khuôn mặt.** Cả hai node còn dùng YuNet, một bộ nhận diện khuôn mặt nhỏ (khoảng 230 KB, giấy phép MIT, từ OpenCV Zoo), để tìm xem người đang ở đâu trước khi phân tích một bức ảnh. Nó được tải một lần, cố định vào một phiên bản xác định của `opencv/face_detection_yunet`, vào `ComfyUI/models/NH-Nodes/yunet/`. Nó chạy trên CPU bằng OpenCV mà NH-Nodes vốn đã cần, nên không tốn VRAM. Nếu đã có sẵn một bản trong thư mục con `yunet` của bất kỳ thư mục `nh_nodes` nào (`models/NH-Nodes/`, hoặc thư mục bạn thêm trong `extra_model_paths.yaml`) thì node dùng luôn bản đó; muốn chạy offline, hãy tự đặt `face_detection_yunet_2023mar.onnx` vào đó. Nếu không nạp được bộ nhận diện (offline ở lần chạy đầu, hoặc OpenCV quá cũ không chạy được nó), node in một cảnh báo trong console và phân tích cả bức ảnh, không có các ảnh cận cảnh nói ở [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh). Hãy khởi động lại ComfyUI để thử lại.

**Bộ nhớ.** Model được đọc từ đĩa một lần và giữ trong bộ nhớ hệ thống. Nó chỉ chuyển sang GPU trong lúc node chạy, rồi trở về bộ nhớ hệ thống, nên không chiếm VRAM giữa các lần chạy. Trước khi chuyển sang GPU, node nhờ ComfyUI gỡ bớt các model của chính nó nếu VRAM đang eo hẹp.

> **Lưu ý:** Nếu node báo nhãn không như mong đợi (unexpected labels), nghĩa là có một checkpoint khác nằm trong một thư mục mà node tìm. Hãy xóa bản đó, model đúng sẽ được tải về.

> **Giấy phép:** Model đứng sau Portrait Segment chỉ được dùng cho nghiên cứu và giáo dục phi thương mại. Model đứng sau Garment Segment không khai báo giấy phép, nên điều khoản của nó chưa rõ. Mã nguồn của NH-Nodes thuộc giấy phép MIT. Bộ nhận diện khuôn mặt YuNet theo giấy phép MIT và không thay đổi điều nào ở trên. Hãy đọc [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) trước khi dùng một trong hai node cho mục đích thương mại.

## Cách các node SegFormer nhìn một bức ảnh

Cả hai model đều nhìn một bức ảnh 512 × 512. Trên ảnh 6000 px, một khuôn mặt rộng 600 px chỉ còn rộng khoảng 50 px trong mắt model, nên các node không đưa cho model cả bức ảnh nguyên khối.

1. **Cả bức ảnh trước.** Ảnh có một cạnh dài hơn gấp đôi cỡ 512 px của model được thu nhỏ bằng cách lấy trung bình theo diện tích (area averaging), nhanh hơn phép resize riêng của model và giữ chi tiết nhỏ sạch sẽ, rồi mới được phân tích. Kết quả đầu tiên này được giữ lại ở mọi chỗ chưa có gì tốt hơn, nên những gì các bước sau bỏ sót vẫn giữ ít nhất kết quả này.
2. **Tìm người.** Bộ nhận diện khuôn mặt tìm khuôn mặt trên hai kích cỡ của bức ảnh. Garment Segment còn lấy thêm những người mà lượt phân tích đầu đã tìm thấy.
3. **Ảnh cận cảnh tinh chỉnh kết quả.** Mỗi khuôn mặt (Portrait Segment) hoặc mỗi nhóm người (Garment Segment) được phân tích lại trên một vùng cắt mà model thấy ở đủ 512 px. Nhãn của vùng cắt được đưa vào kết quả đầu, nhưng không đưa dọc các cạnh cắt của vùng, nơi model thấy vật bị cắt cụt, nên không có cạnh thẳng nào lộ ra trong tóc hay quần áo. Mức độ tin vào một vùng cắt khác nhau tùy node, như các mục Garment Segment và Portrait Segment mô tả.

Với ảnh dài hơn 4096 px, toàn bộ việc trên được làm trên một bản sao dài 4096 px rồi bản đồ nhãn được phóng lên đúng kích thước ảnh ở cuối: model chỉ nhìn 512 px mỗi lần nên nhãn của nó không chứa chi tiết mịn hơn, còn ghép các vùng cắt ở đủ kích thước 8K sẽ mất nhiều giây.

Mỗi node dùng điều này thế nào được nêu trong mục riêng của nó: [Garment Segment](#garment-segment-nh) và [Portrait Segment](#portrait-segment-nh).

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

Tạo mask cho một nhóm trang phục, hoặc tất cả các nhóm cùng lúc, trên những người trong ảnh.

**Dùng khi** bạn muốn có mask sạch của áo, quần, giày hoặc mũ để inpaint, đổi màu hoặc dùng trong workflow thử đồ.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `image` | IMAGE | — | Ảnh chụp một người hoặc một nhóm người. |
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
- **Ảnh cận cảnh của người.** Node phân tích cả bức ảnh trước và giữ kết quả đó. Sau đó nó tìm những người (trong lượt phân tích ấy và bằng bộ nhận diện khuôn mặt) và cắt các vùng vuông, đúng hình dạng đầu vào của model, quanh từng nhóm người: một hàng các vùng cắt chồng lên nhau cho nhóm rộng, một vùng cho một người hoặc một nhóm gọn. Mỗi vùng được phân tích ở đủ 512 px của model, và vùng được nới rộng khi người bị tràn qua một cạnh của nó. Việc này giúp những người nhỏ trong một bức ảnh lớn. Xem [Cách các node SegFormer nhìn một bức ảnh](#cách-các-node-segformer-nhìn-một-bức-ảnh).
- **Vùng cắt chỉ tinh chỉnh.** Nhãn của nó chỉ thay kết quả đầu ở những chỗ model chắc hơn, cách xa cạnh cắt của vùng, và chỉ với các phần nằm trên những người đã tìm thấy. Không có gì bị bịa ra trên một phông trống, và không có cạnh thẳng nào của vùng cắt xẻ qua một món trang phục.

<p align="center"><img src="../assets/examples/garment-segment.jpg" alt="Mask của Garment Segment phủ lên một ảnh mẫu" width="720"><br><em>Garment Segment trên một ảnh mẫu: các vùng tô màu là mask của những phần đã chọn.</em></p>

**Ví dụ**

```text
Load Image → Garment Segment (NH, upper_garment) → mask → Mask Morphology (NH) → inpaint
```

Hãy mở rộng mask thêm vài pixel bằng [Mask Morphology (NH)](./mask.md#mask-morphology-nh) để trang phục được inpaint có thể phủ lên đường viền cũ.

> **Lưu ý:** Model Fashionpedia bỏ sót một số áo, ví dụ chiếc áo cánh nằm dưới một chiếc mũ lớn. Một model trang phục tốt hơn đang được dự kiến.

> **Giấy phép:** model trang phục không khai báo giấy phép. Xem [Trước khi bắt đầu](#trước-khi-bắt-đầu).

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

**Cách hoạt động**

- Các vùng đã bật được gộp thành một mask. Không bật vùng nào thì mask rỗng.
- `face` đã bao gồm mắt, lông mày, mũi, môi và bên trong miệng. Các tùy chọn đó chỉ làm kết quả thay đổi khi `face` tắt.
- Mũ, hoa tai, dây chuyền và quần áo có trong model nhưng node không cung cấp.
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

`NH_SEGS` là dữ liệu mà SAM3 Segment chuyển cho các node Segs. Với mỗi ảnh trong batch (lô ảnh), nó chứa một danh sách instance. Mỗi instance giữ **nhãn** (các từ tiếng Anh mà SAM 3 được yêu cầu tìm: `áo đỏ` được lưu thành `red top`), **điểm tin cậy**, một **khung** ôm sát và mask **chỉ trong phạm vi khung đó**. Mask kích thước đầy đủ chỉ được dựng khi một node xuất `MASK`, nên hàng chục instance trên ảnh 8K vẫn nhẹ.

- Các instance xếp theo thứ tự các nhãn xuất hiện trong concepts của bạn, và theo điểm trong cùng một nhãn.
- Các node Segs xử lý từng ảnh trong batch riêng biệt.
- Các node gốc của ComfyUI không đọc được `NH_SEGS`. Để lấy mask ra, hãy dùng các đầu ra `mask`, [Segs Compose](#segs-compose-nh) hoặc [Segs To Masks](#segs-to-masks-nh). Để dùng detailer của Impact Pack, hãy chuyển đổi bằng [Segs To Impact SEGS](#segs-to-impact-segs-nh).

---

## Viết công thức

`concepts` của SAM3 Segment và `expression` của Segs Compose đọc cùng một ngôn ngữ nhỏ. Bạn gọi tên những thứ mình muốn, ghép chúng bằng các toán tử, và kết quả là một mask cho mỗi ảnh. Trong SAM3 Segment, các từ bạn viết cũng là thứ SAM 3 đi tìm. Trong Segs Compose, chúng chọn các instance đã được tìm sẵn.

| Viết | Nghĩa |
|---|---|
| `person` | Một nhãn: các instance mang tên đó. Không phân biệt chữ hoa/thường. |
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
- **Thứ tự và bản trùng.** Các instance xếp theo thứ tự các nhãn xuất hiện trong công thức (`person` do `background` thêm vào đứng cuối), điểm cao nhất trước trong mỗi nhãn. Hai instance của cùng một nhãn mà mask gần như trùng hoàn toàn (từ 90% trở lên) được tính là một, giữ lại cái điểm cao hơn. Sau đó `max_per_concept` giữ những cái tốt nhất trong phần còn lại. Các instance có điểm đúng bằng nhau giữ thứ tự mà SAM 3 đưa ra.
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
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc một node Segs khác. |
| `labels` | STRING | rỗng | Các nhãn cần giữ, cách nhau bằng dấu phẩy. `*` thay cho một chuỗi ký tự bất kỳ và `?` thay cho một ký tự (`*shirt`). Không phân biệt chữ hoa/thường, và `_` được coi là dấu cách (`red_car` tìm `red car`). Nhãn có dấu cách không cần ngoặc kép ở đây: `wine glass, red car`. Các từ tiếng Việt thông dụng được dịch (`tóc` tìm `hair`); các từ công thức `nền` và `tất cả` không phải nhãn ở đây. Để trống thì giữ mọi nhãn. |
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

Tạo một mask từ nhãn của các instance, bằng chính công thức nhỏ của `concepts` trong SAM3 Segment: hợp, trừ, giao và "mọi thứ trừ...".

**Dùng khi** bạn đã có sẵn các instance và muốn có "phần nền", "người nhưng không có tóc", "bức tường nhưng không có cửa sổ", hoặc một tổ hợp khác của những gì SAM3 Segment tìm được: sau [Segs Select](#segs-select-nh), hoặc để tạo nhiều mask từ một lần chạy SAM 3. Với một mask duy nhất, bạn cũng có thể viết đúng công thức đó thẳng vào `concepts` của SAM3 Segment.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `segs` | NH_SEGS | — | Các instance, từ SAM3 Segment hoặc Segs Select. |
| `expression` | STRING | `all` | Công thức. Xem [Viết công thức](#viết-công-thức). Ô này có nhiều dòng: xuống dòng có tác dụng liệt kê, như dấu phẩy, và nối liền khi nằm cạnh một toán tử. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `mask` | MASK | Kết quả của công thức, một mask cho mỗi ảnh, có kích thước bằng ảnh. |

**Ở đây có gì khác.** Ngôn ngữ chính là ngôn ngữ trong [Viết công thức](#viết-công-thức), với những khác biệt sau so với SAM3 Segment:

- `or`, `and` và `not` là toán tử: `a or b` là `a | b`, `a and b` là `a & b`, `not a` là `~a`.
- Một nhãn chọn các instance mang nhãn đó. Không phân biệt chữ hoa/thường, `_` được coi là dấu cách (`red_car` tìm `red car`), và `*`, `?` là ký tự đại diện (wildcard) (`*shirt`).
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

**Cách hoạt động**

- **Nhãn mà không instance nào có** cho một phần rỗng, và console báo điều đó (`no instance is labelled 'x'; those parts are empty`). Vì thế gõ sai `~persn` sẽ cho cả bức ảnh: hãy xem console khi mask trông lạ.
- **Các từ liền nhau là một nhãn.** Khi không instance nào có nhãn đó nhưng mỗi từ lại là nhãn của một instance (`person hair`), node dừng với một lỗi gợi ý `person + hair`. Với dấu gạch nối cũng vậy: khi cả `person` và `hair` đều có, `person-hair` làm node dừng với lỗi cho thấy `person - hair`.
- **Mỗi ảnh** trong batch có mask riêng, dựng từ chính các instance của nó.
- **Công thức không đọc được** làm node dừng lại với thông báo nêu vị trí của chỗ sai, ví dụ `unexpected '(' at position 8` với `person (adult)`.

<p align="center"><img src="../assets/examples/sam3-segment.jpg" alt="Các instance của SAM3 Segment do Segs Preview vẽ, và ba mask tạo từ chúng bằng Segs Compose và Segs Select trên một ảnh mẫu" width="720"><br><em>Các instance của "person, helmet, face" trên một ảnh mẫu, do Segs Preview vẽ, rồi ba mask tạo từ chúng bằng Segs Compose và Segs Select.</em></p>

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
| Mask của Segs Compose rỗng, và console báo không instance nào có một nhãn | Kiểm tra chính tả của nhãn. Đặt dấu cách quanh dấu trừ: `person - hair`, không phải `person-hair`. |
| Lỗi từ Segs To Masks báo các mask instance kích thước đầy đủ cần nhiều bộ nhớ hơn mức còn trống | Có quá nhiều instance so với kích thước bức ảnh. Dùng Segs Select để chỉ giữ những instance bạn cần, hoặc Segs Compose để lấy một mask duy nhất. |
| Console báo SAM 3 nhìn bức ảnh bị nén vào hình vuông và có thể bỏ sót vật nhỏ | Bức ảnh rộng hoặc cao hơn 4:1. Hãy chia nó thành các ô bằng Image Tile (NH) rồi chạy từng ô. |
| ComfyUI hết VRAM sau khi node chạy | Để `keep_model_loaded` tắt, để trọng số rời GPU sau mỗi lần chạy. |
| Nút Unload Models của ComfyUI không giải phóng VRAM của SAM 3 | `keep_model_loaded` đang bật. Hãy tắt nó và chạy node một lần, hoặc khởi động lại ComfyUI. |

---

[← Về trang hướng dẫn](./README.md)
