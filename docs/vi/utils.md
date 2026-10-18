[← Về trang hướng dẫn](./README.md)

# Utils (tiện ích)

Các node nhỏ giúp workflow gọn gàng: một node có năm thanh trượt, một công tắc bật/tắt điều khiển được nhiều chỗ cùng lúc, và một cặp node gộp tối đa mười dây thành một. Bạn tìm chúng trong menu node tại **NH-Nodes → Utils**; `Pack Universal (NH)` và `Unpack Universal (NH)` nằm ở **NH-Nodes → Utils → Pipe**.

**Nội dung trang**

| Node | Dùng để |
|---|---|
| [Multi-Slider (FLOAT)](#multi-slider-float) | Đặt đến năm số thực từ một node |
| [Multi-Slider (INT)](#multi-slider-int) | Đặt đến năm số nguyên từ một node |
| [Boolean Switch (NH)](#boolean-switch-nh) | Bật/tắt một công tắc có thể điều khiển nhiều chỗ |
| [Pack Universal (NH)](#pack-universal-nh) | Gộp tối đa mười giá trị vào một dây |
| [Unpack Universal (NH)](#unpack-universal-nh) | Lấy một giá trị ra khỏi dây đó |
| [Universal Slider Builder (NH)](#universal-slider-builder-nh) | Đã ngừng dùng (deprecated): hãy dùng các Multi-Slider thay thế |

Trong các bảng, `*` nghĩa là kiểu dữ liệu bất kỳ.

---

## Multi-Slider (FLOAT)

<p align="center"><img src="../assets/nodes/NH_MultiSliderFloat.svg" alt="Multi-Slider (FLOAT) node" width="360"></p>

Năm thanh trượt số thực trong một node, mỗi thanh có một đầu ra riêng.

**Dùng khi** bạn chỉnh nhiều con số, như mức LoRA hay giá trị denoise, và muốn gom chúng vào một bảng điều khiển gọn thay vì năm node riêng.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `float_1` … `float_5` | `FLOAT` | `0.0` | Năm thanh trượt, mỗi thanh từ -1.0 đến 1.0, bước 0.01. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `Float 1` … `Float 5` | `FLOAT` | Giá trị của thanh trượt có cùng số. |

**Ví dụ — một bảng điều khiển cho LoRA và sampler**

```text
Multi-Slider (FLOAT) → Load LoRA Model Index (NH) / KSampler
```

Nối `Float 1` vào `strength_model` của loader LoRA và `Float 2` vào `denoise` của `KSampler`. Giờ bạn chỉnh cả hai từ một node.

> **Mẹo:** Phạm vi là -1.0 đến 1.0. Với số nguyên lớn hơn, hãy dùng `Multi-Slider (INT)`, hoặc gõ giá trị trực tiếp vào node đích.

---

## Multi-Slider (INT)

<p align="center"><img src="../assets/nodes/NH_MultiSliderInt.svg" alt="Multi-Slider (INT) node" width="360"></p>

Năm thanh trượt số nguyên trong một node, mỗi thanh có một đầu ra riêng.

**Dùng khi** bạn đặt nhiều số nguyên, như số slot hay số lượng, từ một bảng điều khiển.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `int_1` … `int_5` | `INT` | `0` | Năm thanh trượt, mỗi thanh từ -100 đến 100, bước 1. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `Int 1` … `Int 5` | `INT` | Giá trị của thanh trượt có cùng số. |

**Ví dụ — chọn LoRA và VAE chỉ với hai thanh trượt**

```text
Multi-Slider (INT) → Load LoRA Model Index (NH) [index]
Multi-Slider (INT) → Load VAE Index (NH) [index]
```

Nối `Int 1` vào `index` của loader LoRA và `Int 2` vào `index` của loader VAE. Slot đếm từ 1, còn các thanh trượt bắt đầu ở `0`, nên hãy kéo chúng lên 1 trở lên trước khi chạy. Giá trị nhỏ hơn 1 không chọn gì và chặn loader (xem [Loaders](./loaders.md)).

---

## Boolean Switch (NH)

<p align="center"><img src="../assets/nodes/NH_BooleanSwitch.svg" alt="Boolean Switch (NH) node" width="360"></p>

Một công tắc bật/tắt trả về giá trị `BOOLEAN`.

**Dùng khi** một thiết lập cần điều khiển nhiều chỗ cùng lúc, ví dụ `condition` của `If/Else (NH)`, đầu vào `enabled` của `Gate Switch (NH)`, hoặc `reset` của `Counter (NH)`.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `boolean_switch` | `BOOLEAN` | `true` | Công tắc. Hiển thị là On (True) hoặc Off (False). |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `boolean` | `BOOLEAN` | Trạng thái của công tắc. |

**Ví dụ — nhanh hay chất lượng, từ một công tắc**

```text
Boolean Switch (NH) → If/Else (NH)
```

Nối `boolean` vào `condition`. Đặt nhánh nhanh vào `if_true` và nhánh chất lượng vào `if_false`. Chỉ nhánh được chọn mới chạy. Xem [Logic](./logic.md).

---

## Pack Universal (NH)

<p align="center"><img src="../assets/nodes/NH_PackUniversal.svg" alt="Pack Universal (NH) node" width="360"></p>

Gộp tối đa mười giá trị thuộc bất kỳ kiểu nào vào một dây duy nhất, kiểu `NH_UNIVERSAL_PIPE`.

**Dùng khi** nhiều dây (model, CLIP, VAE, prompt, số) cùng đi đến một chỗ xa trong workflow và một dây duy nhất sẽ giữ canvas gọn gàng.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `input_0` … `input_9` | `*` | chưa nối | Tùy chọn. Giá trị cần lưu. Mỗi giá trị được lưu ở vị trí có cùng số. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `package` | `NH_UNIVERSAL_PIPE` | Pipe chứa tất cả các giá trị. |

**Cách hoạt động**

- Pipe luôn có 10 vị trí, đánh số từ 0 đến 9. `input_N` đi vào vị trí N.
- Vị trí bạn không nối vẫn để trống. Nó không bao giờ làm lệch các vị trí khác: nếu bạn chỉ nối `input_1` và `input_3`, chúng vẫn nằm ở vị trí 1 và 3.
- Không có bước kiểm tra kiểu dữ liệu. Pack nhận mọi kiểu, và `Unpack Universal (NH)` trả về mọi kiểu.

---

## Unpack Universal (NH)

<p align="center"><img src="../assets/nodes/NH_UnpackUniversal.svg" alt="Unpack Universal (NH) node" width="360"></p>

Lấy một giá trị ra khỏi pipe theo số vị trí.

**Dùng khi** bạn đã có một pipe từ `Pack Universal (NH)` và cần lấy một giá trị trong đó.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `index` | `INT` | `0` | Vị trí cần đọc, từ 0 đến 9. Dùng số của `input_N` mà bạn đã nối trên Pack. |
| `package` | `NH_UNIVERSAL_PIPE` | — | Tùy chọn. Pipe cần đọc. |
| `pipe` | `NH_UNIVERSAL_PIPE` | — | Tùy chọn. Đầu vào pipe thứ hai. Chỉ được dùng khi `package` chưa được nối. |

| Đầu ra | Kiểu | Mô tả |
|---|---|---|
| `item` | `*` | Giá trị tại vị trí đó. |

**Cách hoạt động**

- `item` rỗng (`None`) khi vị trí không chứa gì, khi `index` nằm ngoài khoảng 0 đến 9, hoặc khi chưa nối pipe nào. Một node cần giá trị thật thường sẽ dừng với lỗi, nên hãy kiểm tra số vị trí trước.
- Mỗi `Unpack Universal (NH)` trả về một giá trị. Hãy thêm một node cho mỗi giá trị bạn cần.

**Ví dụ — đưa cả bộ model qua một dây**

```text
Load Checkpoint → Pack Universal (NH) → Unpack Universal (NH) → KSampler
```

Hãy lên kế hoạch các vị trí một lần, rồi dùng cùng số ở cả hai phía:

| Trên Pack | Mang | Trên Unpack (`index`) |
|---|---|---|
| `input_0` | `MODEL` | `0` |
| `input_1` | `CLIP` | `1` |
| `input_2` | `VAE` | `2` |
| `input_3` | positive prompt (`STRING`) | `3` |

> **Mẹo:** Hãy đặt tiêu đề rõ ràng cho mỗi cặp Pack và Unpack, và ghi bảng kế hoạch ở trên vào một note đặt cạnh chúng.

---

## Universal Slider Builder (NH)

<p align="center"><img src="../assets/nodes/NH_UniversalSliderBuilder.svg" alt="Universal Slider Builder (NH) node" width="360"></p>

**Đã ngừng dùng (deprecated).** Node này bị ẩn khỏi ô tìm kiếm node và chỉ được giữ lại để các workflow cũ có chứa nó vẫn mở được. Nó chưa bao giờ tạo ra đầu ra, nên không dùng được cho việc gì. Hãy thay nó bằng `Multi-Slider (FLOAT)` hoặc `Multi-Slider (INT)`.

| Đầu vào | Kiểu | Mặc định | Mô tả |
|---|---|---|---|
| `config` | `STRING` | hai dòng ví dụ | Một ô văn bản nhiều dòng. Mỗi dòng vốn dùng để định nghĩa một thanh trượt (`name, type, default, min, max, step`), nhưng node chưa bao giờ tạo ra đầu ra, nên nội dung này không ảnh hưởng gì đến workflow. |

---

**Xem thêm:** [Loaders](./loaders.md) cho các node mà những thanh trượt và công tắc này thường điều khiển, [Batch & Lists](./batch-lists.md) cho `Counter (NH)`, và [Logic](./logic.md) cho `If/Else (NH)` và `Gate Switch (NH)`.

[← Về trang hướng dẫn](./README.md)
