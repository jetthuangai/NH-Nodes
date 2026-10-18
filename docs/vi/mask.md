[← Về trang hướng dẫn](./README.md)

# Mask

Bốn node nhỏ để làm việc với mask: mở rộng hoặc làm mềm mask, đo nó, biến nó thành hộp chữ nhật, và cho hộp của mask này cùng tỉ lệ với hộp của mask khác. Bạn tìm thấy chúng trong **NH-Nodes → Mask**.

Một `MASK` là một batch (lô ảnh) một kênh. `1.0` là bên trong mask, `0.0` là bên ngoài, và các giá trị ở giữa là cạnh mềm. Các node đo hoặc đóng hộp một mask coi một điểm ảnh là nằm trong khi giá trị của nó lớn hơn `0.5`.

**Trên trang này**

- [Mask Morphology (NH)](#mask-morphology-nh) — mở rộng, thu hẹp, lấp lỗ và làm mềm mask
- [Mask Properties (NH)](#mask-properties-nh) — kích thước và tọa độ các góc của hộp bao mask
- [NH Create Box Mask](#nh-create-box-mask) — thay mask bằng hình chữ nhật bao quanh nó
- [NH Mask Aspect Ratio Match](#nh-mask-aspect-ratio-match) — cho hộp của mask này cùng tỉ lệ khung hình với mask khác

---

## Mask Morphology (NH)

<p align="center"><img src="../assets/nodes/NH_MaskAdvancedMorphology.svg" alt="Mask Morphology (NH) node" width="360"></p>

Mở rộng hoặc thu hẹp mask theo từng trục, lấp các lỗ bên trong và làm mềm cạnh.

**Dùng khi** mask phân đoạn quá sát nên cần chừa lề để inpaint, khi mask có lỗ bạn muốn lấp, hoặc khi cạnh cứng của nó cần mờ dần cho mềm.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `mask` | MASK | — | Mask hoặc batch mask. Một mask 2-D đơn lẻ cũng được chấp nhận, coi như batch một ảnh. |
| `horizontal_expand` | INT | `0` | Khoảng -512 đến 512. Giá trị dương mở rộng mask sang trái và phải, giá trị âm thu hẹp nó. Giá trị là độ dài kernel tính bằng pixel: mỗi cạnh dịch đi một nửa giá trị đó, làm tròn xuống. |
| `vertical_expand` | INT | `0` | Khoảng -512 đến 512. Tương tự, cho cạnh trên và cạnh dưới. |
| `fill_holes` | BOOLEAN | `false` | Lấp các lỗ nằm trọn bên trong mask. Điểm ảnh lớn hơn `0.5` được coi là nằm trong. |
| `blur_radius` | FLOAT | `0.0` | Khoảng 0 đến 100, bước 0.1. Độ mạnh của làm mờ Gaussian, tính bằng pixel. `0` giữ nguyên cạnh. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `MASK` | MASK | Batch mask đã xử lý, cùng kích thước với đầu vào. |

**Cách hoạt động**

- **Thứ tự:** ngang, rồi dọc, rồi lấp lỗ, rồi làm mờ.
- **Độ dài kernel:** độ dài chẵn được làm tròn lên số lẻ kế tiếp để kết quả vẫn cân đối quanh tâm. Bảng bên dưới cho thấy mỗi cạnh dịch bao nhiêu.
- **Mask mềm vẫn mềm.** Mở rộng lấy giá trị sáng nhất lân cận còn thu hẹp lấy giá trị tối nhất, nên dải chuyển sắc được giữ và không giá trị nào bị ép về 0 hay 1.
- **Mép ảnh không ăn mất mask.** Hình chạm mép ảnh vẫn chạm mép đó khi bạn thu hẹp.
- Việc mở rộng và thu hẹp chạy dưới dạng phép toán PyTorch trên thiết bị đang chứa mask (CPU hoặc GPU), nên mask lớn và batch mask dài được xử lý hiệu quả. Lấp lỗ và làm mờ chạy trên CPU, từng mask một.

| `horizontal_expand` hoặc `vertical_expand` | Mỗi cạnh dịch |
|---|---|
| `0`, `1`, `-1` | 0 px, không đổi |
| `2`, `3` (hoặc số âm tương ứng) | 1 px |
| `4`, `5` (hoặc số âm tương ứng) | 2 px |
| `10`, `11` (hoặc số âm tương ứng) | 5 px |

<p align="center"><img src="../assets/examples/mask-morphology.jpg" alt="Một mask được mở rộng, thu hẹp, mở và đóng bằng Mask Morphology" width="720"><br><em>Mở rộng, thu hẹp, và hai node nối tiếp nhau: thu hẹp rồi mở rộng (phép mở, open), mở rộng rồi thu hẹp (phép đóng, close).</em></p>

**Ví dụ**

```text
Garment Segment (NH) → Mask Morphology (NH) → inpaint
```

Với `horizontal_expand` `17`, `vertical_expand` `17`, bật `fill_holes` và `blur_radius` `4`, mỗi cạnh nở ra 8 px, các lỗ được lấp và cạnh mờ dần mềm mại.

> **Lưu ý:** `fill_holes` làm mask trở nên cứng trước: mọi điểm ảnh chỉ còn là `0` hoặc `1` trước khi làm mờ. Hãy tắt nó nếu bạn cần giữ dải chuyển sắc mềm sẵn có.

> **Mẹo:** Việc mở rộng dùng kernel hình chữ nhật nên các góc vẫn vuông. Thêm một `blur_radius` nhỏ để làm mềm chúng.

---

## Mask Properties (NH)

<p align="center"><img src="../assets/nodes/NH_MaskProperties.svg" alt="Mask Properties (NH) node" width="360"></p>

Đo hộp bao (bounding box) của mask: kích thước và tọa độ các góc.

**Dùng khi** bạn cần con số cho vùng mà mask phủ, ví dụ để cắt (crop) ảnh quanh chủ thể hoặc chuyển kích thước hộp cho node khác.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `mask` | MASK | — | Mask cần đo. Chỉ mask đầu tiên của batch được đo. Điểm ảnh lớn hơn `0.5` được coi là nằm trong. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `width` | INT | Chiều rộng hộp tính bằng pixel, `x2 - x1 + 1`. |
| `height` | INT | Chiều cao hộp tính bằng pixel, `y2 - y1 + 1`. |
| `x1` | INT | Cạnh trái: cột đầu tiên nằm trong mask. |
| `y1` | INT | Cạnh trên: hàng đầu tiên nằm trong mask. |
| `x2` | INT | Cạnh phải: cột cuối cùng nằm trong mask. |
| `y2` | INT | Cạnh dưới: hàng cuối cùng nằm trong mask. |
| `bbox` | BBOX | Cùng chiếc hộp đó dưới dạng một giá trị, `(x1, y1, x2, y2)`. |

**Cách hoạt động**

- Cả hai đầu đều được tính vào (inclusive). Mask phủ cột 10 đến 29 và hàng 5 đến 24 cho `x1` 10, `x2` 29, `y1` 5, `y2` 24, `width` 20 và `height` 20.
- Mask rỗng (không có điểm ảnh nào lớn hơn `0.5`) cho `0` ở mọi con số và `(0, 0, 0, 0)` ở `bbox`.

**Ví dụ**

```text
Garment Segment (NH) → Mask Properties (NH) → x1, y1, width, height → node cắt ảnh
```

---

## NH Create Box Mask

<p align="center"><img src="../assets/nodes/NH_MaskToBBox.svg" alt="NH Create Box Mask node" width="360"></p>

Thay mỗi mask bằng hình chữ nhật đặc bao sát quanh nó.

**Dùng khi** bạn muốn một vùng inpaint hình chữ nhật quanh chủ thể thay vì đường viền chính xác của nó.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `mask` | MASK | — | Mask hoặc batch mask. Điểm ảnh lớn hơn `0.5` được coi là nằm trong. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `box_mask` | MASK | Một hình chữ nhật `1.0` cho mỗi mask đầu vào, phủ hộp bao của nó. Mask rỗng cho ra mask rỗng (toàn `0.0`). |

**Cách hoạt động**

- Mỗi mask trong batch được đóng hộp riêng.
- Hộp bao sát. Muốn thêm lề, hãy cho `box_mask` đi qua [Mask Morphology (NH)](#mask-morphology-nh) và mở rộng nó.

**Ví dụ**

```text
Garment Segment (NH) → NH Create Box Mask → Mask Morphology (NH, mở rộng) → inpaint
```

---

## NH Mask Aspect Ratio Match

<p align="center"><img src="../assets/nodes/NH_MaskAspectRatioMatch.svg" alt="NH Mask Aspect Ratio Match node" width="360"></p>

Định hình lại hộp bao của một mask để nó có cùng tỉ lệ khung hình với hộp bao của mask khác.

**Dùng khi** một hộp mask phải theo tỉ lệ của hình tham chiếu, ví dụ một hộp vuông cần khớp tỉ lệ với món trang phục trước khi bạn inpaint hoặc cắt.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `target_ratio_mask` | MASK | — | Mask có hộp bao đặt ra tỉ lệ khung hình cần hướng tới. Chỉ hộp bao của nó có ý nghĩa. |
| `mask_to_adjust` | MASK | — | Mask cần định hình lại. |
| `mode` | COMBO | `stretch` | Cách đạt tỉ lệ: `stretch`, `pad` hoặc `crop`. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `MASK` | MASK | Batch mask đã định hình lại. Mọi thứ bên ngoài hộp mới là `0.0`. |

**Các chế độ**

| Chế độ | Việc nó làm |
|---|---|
| `stretch` | Phóng nội dung cho vừa hộp mới, không làm mượt (lấy mẫu nearest). |
| `pad` | Giữ nội dung tại chỗ và thêm điểm ảnh mask (`1.0`) quanh nó cho đến khi hộp đạt tỉ lệ. |
| `crop` | Chỉ giữ phần giữa của nội dung để hộp đạt tỉ lệ. Phần còn lại không bị dịch chỗ. |

**Cách hoạt động**

- Node chỉ thay đổi một cạnh, quanh tâm hộp của `mask_to_adjust`: `stretch` và `pad` làm hộp dài thêm theo một trục, `crop` làm hộp ngắn đi.
- Hộp nào sẽ vượt ra ngoài ảnh thì được trượt ngược vào trong.
- Mọi mask trong batch đều được xử lý. Nếu `target_ratio_mask` có ít mask hơn thì chúng lặp vòng.
- `target_ratio_mask` rỗng được coi là 1:1. `mask_to_adjust` rỗng được trả về nguyên vẹn.

Với hộp đích 100 × 50 (2:1) và hộp cần chỉnh 40 × 40:

| `mode` | Hộp mới | Kết quả |
|---|---|---|
| `stretch` | 80 × 40 | Nội dung 40 × 40 được phóng thành 80 × 40. |
| `pad` | 80 × 40 | Nội dung 40 × 40 giữ nguyên, thêm 20 px mask ở bên trái và bên phải. |
| `crop` | 40 × 20 | Giữ lại phần giữa 40 × 20 của nội dung. |

<p align="center"><img src="../assets/examples/mask-aspect.jpg" alt="Mask Aspect Ratio Match ở chế độ pad, crop và stretch" width="720"><br><em>Cùng một mask được khớp theo tỉ lệ đích bằng <code>pad</code>, <code>crop</code> và <code>stretch</code>.</em></p>

**Ví dụ**

```text
mask của Garment Segment (NH) → target_ratio_mask ┐
                                                  ├→ NH Mask Aspect Ratio Match (pad) → mask inpaint
mask hộp → mask_to_adjust ────────────────────────┘
```

> **Mẹo:** Dùng `pad` khi vùng phải được phủ kín, và `crop` khi bạn muốn một lõi nhỏ hơn nhưng đúng tỉ lệ.

---

[← Về trang hướng dẫn](./README.md)
