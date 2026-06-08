# Ke hoach: NH Large Image Preview Nodes

## Boi canh

Mot so workflow upscale / generate anh lon co the lam UI ComfyUI bi giat sau khi chay xong, dac biet khi node preview hoac compare dang hien thi anh full-res truc tiep tren canvas. Vi du workflow upscale ra anh khoang 4608 x 6912 co the tao temp PNG rat lon, khien trinh duyet phai decode va ve lai bitmap nang moi khi pan, zoom, select node.

Muc tieu cua bo node moi trong NH-Nodes la giu trai nghiem lam viec muot ma, nhung van cho phep xem anh lon voi chat luong cao khi can.

## Nguyen tac thiet ke

- Canvas cua ComfyUI chi nen hien thumbnail / proxy nhe.
- Anh full-res khong duoc render truc tiep trong vong repaint cua LiteGraph canvas.
- Khi user can xem chi tiet, mo viewer rieng bang modal, side panel, hoac tab noi bo.
- Viewer anh lon nen dung tile pyramid / deep zoom de chi load vung dang xem.
- Khong lam giam chat luong output goc; chi giam kich thuoc ban preview tren canvas.
- Metadata nhe duoc luu trong workflow; tranh luu danh sach temp URL full-res cu trong widget.

## Bo node de xuat

### 1. NH Large Image Preview

Node preview chinh de thay the PreviewImage trong cac workflow tao anh lon.

Input:

- `image`: IMAGE

Option de xuat:

- `preview_megapixels`: gioi han kich thuoc thumbnail/proxy tren canvas, mac dinh 1-2 MP.
- `tile_size`: kich thuoc tile, mac dinh 512.
- `tile_format`: `jpg` hoac `webp`.
- `tile_quality`: mac dinh 85-92.
- `viewer_mode`: `modal`, `side_panel`, hoac `new_tab`.
- `cache_policy`: `temp`, `session`, hoac `persistent`.
- `title`: label hien thi trong viewer.

Output:

- `image`: IMAGE passthrough de workflow tiep tuc noi node.
- `manifest`: STRING hoac metadata dict neu can dung cho node khac.

Hanh vi:

- Khi execute, tao thumbnail nhe va manifest.
- Neu anh lon hon nguong, tao tile pyramid trong thu muc cache.
- UI node chi hien thumbnail/proxy va nut mo viewer.
- Viewer full-res load tile theo viewport, khong ep canvas ComfyUI ve full bitmap.

### 2. NH Large Image Compare

Node so sanh anh lon, thay cho comparer full-res tren canvas.

Input:

- `image_a`: IMAGE
- `image_b`: IMAGE

Option de xuat:

- `compare_mode`: slider, side-by-side, difference.
- `sync_pan_zoom`: mac dinh bat.
- `preview_megapixels`: kich thuoc proxy tren node.
- `tile_size`, `tile_format`, `tile_quality`.

Hanh vi:

- Tao tile pyramid rieng cho A va B.
- Canvas chi hien thumbnail compare nhe.
- Viewer rieng co slider truot A/B, pan/zoom dong bo, fit/100% zoom.
- Khong luu temp image full-res vao widget cua node.

### 3. NH Save Large Image

Node save anh full-res ma khong gan preview nang vao canvas.

Input:

- `image`: IMAGE

Option de xuat:

- `filename_prefix`
- `format`: png, jpg, webp, tiff.
- `quality`
- `embed_metadata`
- `open_after_save`: mac dinh tat.

Output:

- `image`: IMAGE passthrough.
- `file_path`: STRING.

Hanh vi:

- Luu output goc chat luong cao.
- Tra ve path de node preview/viewer co the tham chieu neu can.
- Khong mac dinh hien full-res tren canvas.

### 4. NH Large Preview Cache Manager

Node hoac module quan ly cache cho cac node preview lon.

Chuc nang:

- Xoa cache theo tuoi file.
- Gioi han tong dung luong cache.
- Xoa cache theo cache id.
- Hien thong tin dung luong cache hien tai.

Co the bat dau bang backend utility thay vi node rieng, sau do them node neu can UX ro rang hon.

## Kien truc backend

De xuat them module moi trong NH-Nodes:

- `large_image_preview.py`: node definitions va logic chinh.
- `large_preview_cache.py`: tao cache id, ghi thumbnail, ghi tile, cleanup.
- `large_preview_routes.py`: dang ky HTTP routes cho ComfyUI frontend.

Luon uu tien dung API / pattern hien co cua ComfyUI va NH-Nodes.

### Cache layout

Vi du:

```text
ComfyUI/temp/nh_large_preview/
  <cache_id>/
    manifest.json
    thumb.jpg
    level_0/
      0_0.jpg
    level_1/
      0_0.jpg
      1_0.jpg
```

Manifest de xuat:

```json
{
  "cache_id": "abc123",
  "width": 4608,
  "height": 6912,
  "tile_size": 512,
  "format": "jpg",
  "levels": [
    { "level": 0, "width": 288, "height": 432 },
    { "level": 1, "width": 576, "height": 864 }
  ],
  "thumb_url": "/nh-nodes/large-preview/thumb/abc123",
  "tile_url_template": "/nh-nodes/large-preview/tile/abc123/{level}/{x}_{y}.jpg"
}
```

### HTTP routes

Dang ky bang `PromptServer.instance.routes`:

- `GET /nh-nodes/large-preview/manifest/{cache_id}`
- `GET /nh-nodes/large-preview/thumb/{cache_id}`
- `GET /nh-nodes/large-preview/tile/{cache_id}/{level}/{tile}`
- `POST /nh-nodes/large-preview/cleanup`

Can validate `cache_id`, `level`, `tile` de tranh path traversal.

### Tao tile pyramid

Lua chon uu tien:

- `pyvips` / libvips neu cai duoc on dinh tren Windows: nhanh, tiet kiem RAM, hop voi anh rat lon.

Fallback:

- Pillow / PIL: don gian hon, it phu thuoc hon, chap nhan cham hon cho phase dau.

Khuyen nghi:

- Khong bien `pyvips` thanh hard dependency ngay tu phase 1.
- Neu khong co `pyvips`, node van chay bang Pillow va hien warning nhe.
- Nen bat dau bang thumbnail-only de giai quyet lag UI nhanh, sau do them tile pyramid.

## Kien truc frontend

Them file:

- `web/nh_large_image_preview.js`

Dung pattern `app.registerExtension` nhu cac extension hien co cua NH-Nodes.

Hanh vi UI:

- Node nhan metadata tu backend sau khi execute.
- Node ve thumbnail nhe hoac hien widget image nho.
- Nut `Open` mo viewer rieng.
- Viewer nam ngoai LiteGraph canvas repaint loop.
- Viewer co cac control co ban: fit, 100%, zoom in/out, download/open file, copy path neu co.

Thu vien viewer:

- Uu tien OpenSeadragon neu can deep zoom day du.
- Co the bundle vao `web/vendor/openseadragon/` de tranh phu thuoc CDN.
- Neu muon toi gian, phase dau co the viet tile viewer canvas custom, nhung OpenSeadragon se nhanh va it rui ro hon.

## Phases de build

### Phase 1: Thumbnail-only preview

Muc tieu:

- Tao `NH Large Image Preview` thay the PreviewImage cho workflow anh lon.
- Backend tao thumbnail/proxy gioi han 1-2 MP.
- Node canvas chi hien thumbnail.
- Nut mo file/temp image co the tam thoi dung URL anh full-res neu chua co tile viewer.

Acceptance:

- Workflow anh khoang 4608 x 6912 generate xong van pan/drag UI muot hon ro.
- Canvas khong render full-res bitmap.

### Phase 2: Tile pyramid viewer

Muc tieu:

- Tao tile pyramid va manifest.
- Them viewer modal/side panel dung tile loading.
- Ho tro fit, zoom, pan, 100%.

Acceptance:

- Xem anh full-res lon van muot.
- Viewer chi load tile can thiet theo viewport.
- Browser refresh van mo lai duoc neu cache con ton tai.

### Phase 3: Large Image Compare

Muc tieu:

- Tao `NH Large Image Compare`.
- So sanh A/B bang tiled viewer.
- Ho tro slider va sync pan/zoom.

Acceptance:

- So sanh 2 anh lon khong lam canvas giat.
- Slider muot, khong decode full 2 bitmap lon tren canvas.

### Phase 4: Cache manager va polish

Muc tieu:

- Gioi han cache theo dung luong va tuoi file.
- Them cleanup route / command / node neu can.
- Viet README docs ngan va workflow mau.

Acceptance:

- Cache khong tang vo han.
- Thong bao loi ro rang khi cache bi xoa.
- Workflow mau chay duoc sau khi cai node.

## Test va benchmark

Workflow regression:

- Dung workflow `utility_pid_latent_upscale_dit.json` lam case chinh vi da tung gay lag khi preview full-res.

Kich thuoc test:

- 2 MP: case nhe.
- 8 MP: case trung binh.
- 32 MP: case lon tuong tu workflow upscale.

Can kiem tra:

- Sau khi queue xong, pan/drag/select node tren canvas co muot khong.
- Browser memory co tang bat thuong khong.
- Thumbnail co hien dung va khong bi meo aspect ratio.
- Tile viewer zoom/pan dung, khong load thua qua nhieu tile.
- Cache cleanup khong xoa nham file ngoai thu muc cache.
- Workflow luu/mo lai khong mang theo temp URL nang khong can thiet.

## Rui ro

- `pyvips` tren Windows co the kho cai; can fallback Pillow.
- Viewer frontend can dong bo voi cach ComfyUI phat event node executed.
- Cache route phai validate path can than de tranh doc file ngoai thu muc cache.
- Tao tile cho anh rat lon co the ton CPU/IO; can lam sau khi generate xong va co threshold hop ly.
- Neu node vua save full image vua tao tile, can tranh ghi duplicate qua nhieu.

## Non-goals

- Khong thay the toan bo SaveImage cua ComfyUI.
- Khong toi uu toc do inference/GPU.
- Khong render full-res bitmap truc tiep tren LiteGraph canvas.
- Khong bat buoc moi workflow phai dung viewer moi; day la node tu chon cho workflow anh lon.

## Cau hoi can chot truoc khi build

- Viewer nen mac dinh la modal, side panel, hay tab rieng?
- Mac dinh thumbnail/proxy nen la 1 MP, 2 MP, hay theo do phan giai man hinh?
- Co can node compare ngay phase dau khong, hay lam preview truoc?
- Cache nen nam trong `temp` hay co option luu persistent trong `output`?
- Co chap nhan bundle OpenSeadragon vao repo khong, hay muon viet viewer toi gian khong dependency?

## De xuat uu tien

Nen build theo thu tu:

1. `NH Large Image Preview` thumbnail-only.
2. Tile pyramid + OpenSeadragon viewer.
3. `NH Large Image Compare`.
4. Cache manager va workflow mau.

Thu tu nay giai quyet ngay van de UI lag, sau do moi tang dan kha nang xem full-res va compare nang cao.
