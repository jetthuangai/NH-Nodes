[← Về trang hướng dẫn](./README.md)

# Vision (phân đoạn ảnh)

Hai node cắt từng phần trên ảnh chụp một người bằng các model SegFormer: trang phục với Garment Segment, và khuôn mặt, tóc, cổ với Portrait Segment. Mỗi node trả về một mask có cùng kích thước với ảnh đầu vào, một ảnh cắt nền có độ trong suốt và một ảnh xem trước mask. Bạn tìm thấy chúng trong **NH-Nodes → Vision**.

**Trên trang này**

- [Trước khi bắt đầu](#trước-khi-bắt-đầu) — model, lần chạy đầu tiên, giấy phép, bộ nhớ
- [Tinh chỉnh mask](#tinh-chỉnh-mask) — các tùy chọn dùng chung của hai node
- [Garment Segment (NH)](#garment-segment-nh) — mask trang phục thân trên, thân dưới, giày dép hoặc mũ nón
- [Portrait Segment (NH)](#portrait-segment-nh) — mask khuôn mặt, tóc, cổ và các bộ phận trên mặt

## Trước khi bắt đầu

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

Cả hai node biến bản đồ nhãn của model thành mask, rồi tinh chỉnh theo thứ tự: `fill_holes`, `mask_expand`, `mask_blur`. Mask luôn có kích thước bằng ảnh đầu vào.

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

[← Về trang hướng dẫn](./README.md)
