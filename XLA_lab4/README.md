# **NHẬP MÔN XỬ LÍ ẢNH SỐ - LAB4**
## Phân Vùng Ảnh


**Sinh viên thực hiện: Võ Hoàng Hải**

**MSSV: 2374802010703** 

**Môn học: Nhập môn xử lí ảnh số** 

**Giảng viên: Đỗ Hữu Quân** 



## Giới thiệu

Bài lab này nhằm mục đích thực hiện **việc phân vùng và xử lí hình ảnh** 


## Công nghệ sử dụng


- **Python**: Ngôn ngữ chính                           
- **Pillow (PIL)**: Đọc, chuyển đổi, và lưu ảnh              
- **NumPy**: Xử lý ảnh dưới dạng mảng số học          
- **ImageIO**: Đọc file ảnh với định dạng hiện đại      
- **Matplotlib**: Hiển thị ảnh trực quan


## Chi tiết các phép phân vùng và công thức

### 1. Cài đặt thư viện
**Mục đích:**
- xử lí ảnh: làm mờ, lọc nhiễu, biến đổi hình học
- nhận diện đối tượng: khuôn mặt, vật thể

**Code chính**
```python
pip install opencv-python
```
---
### 2. Phân tích vùng ảnh
#### 2.1. Phân vùng theo histogram
##### 2.1.1. Phương pháp Otsu
**Mục đích:**
- Tự động tìm giá trị ngưỡng tối ưu để tách ảnh xám thành ảnh nhị phân
- Giúp phân biệt rõ đối tượng và nền
- Ứng dụng trong nhận dạng ảnh, xử lí ảnh y tế, OCR, tách vật thể

**Công thức toán học:**
  ```math
  \sigma_b^2(t) = \omega_0(t) \cdot \omega_1(t) \cdot \left[\mu_0(t) - \mu_1(t)\right]^2
  ```
  -  Phương sai giữa hai lớp tại ngưỡng: `t`
    
**Ví dụ:**
- Ảnh chứa chữ đen trên nền trắng → Otsu tự động chọn ngưỡng phân tách rõ nét chữ và nền giấy.
  
**Code chính:**
  
  ```python
  a = np.asarray(data)
  thres = threshold_otsu(a)
  b = a > thres
  b = Image.fromarray(b)
  ```
  ---
##### 2.1.2. Phương pháp Adaptive Thresholding

**Mục dích:**
- Khắc phục nhược điểm của phương pháp ngưỡng hóa toàn cục (global thresholding như Otsu) trong các ảnh có ánh sáng không đồng đều
- Giúp tách đối tượng/nền chính xác hơn trong điều kiện sáng–tối không đều
- Ứng dụng trong xử lý ảnh tài liệu, nhận dạng văn bản, ảnh scan mờ
  
**Công thức toán học:**
  
  ```math
  T(x, y) = \mu(x, y) - C
  ```
  - `c`: Hằng số điều chỉnh giúp giảm nhiễu
  - `T(x,y)`:ngưỡng tại điểm ảnh `(x,y)`

**Ví dụ:**
- Ảnh tài liệu chụp từ camera, có bóng tối hoặc ánh sáng loá → phương pháp này vẫn phân biệt được chữ rõ nét.

**Code chính:**
```python
data=Image.open('fruit.jfif').convert('L')
a = np.asarray(data)
b=threshold_local(a, 39, offset=10)
b= Image.fromarray(b)
```
---

#### 2.2 Phân vùng theo region
**mục đích:**
- Phân chia ảnh thành các vùng có đặc điểm giống nhau (mức xám, màu, kết cấu)
- Tách các đối tượng bằng cách dựa vào sự đồng nhất bên trong mỗi vùng
- Phù hợp với ảnh có vùng tương đối đồng đều, ví dụ ảnh y tế, ảnh vật thể có nền rõ ràng

**Công thức toán học:**
- region growing:
  
  nếu:
  ```math
  |I(x, y) − I_seed| < T
  ```
  thì điểm ảnh `(x,y)` được thêm vào vùng
  - `I(x, y)`: giá trị mức xám tại điểm ảnh `(x, y)`
  - ```I_seed```: giá trị mức xám của điểm hạt giống
  - `T`: ngưỡng cho phép (threshold)
  - Nếu điều kiện đúng → `(x, y)` được đưa vào vùng

- region splitting and merging:
  - điều kiện gộp
  ```math
  |I₁ − I₂| < T
  ```
  - `I₁`, `I₂`: giá trị trung bình mức xám của 2 vùng liền kề
  - `T`: ngưỡng cho phép

**Ví dụ:**
- Ảnh CT scan: phân vùng mô mềm, xương, khối u
- Tách vùng đồng màu trong ảnh thiên nhiên (đất, nước, rừng)

**Code chính:**
```python
data=cv2.imread('fruit1.jpg')
a=cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
thresh, b1 = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
b2=cv2.erode(b1, None, iterations = 2)
dist_trans = cv2.distanceTransform(b2, 2,3)
thresh, dt = cv2.threshold(dist_trans, 1,255,cv2.THRESH_BINARY)
labelled,ncc=label(dt)
labelled = labelled.astype(np.int32)
cv2.watershed(data, labelled)
b=Image.fromarray(labelled)
```
---

#### 2.3. Biến đổi đối tượng trong ảnh
##### 2.3.1. sử dụng binary_dilation

**Mục đích:**
- Mở rộng vùng trắng (giá trị 1) trong ảnh nhị phân
- Làm đầy các lỗ nhỏ, nối các vùng trắng gần nhau
- Ứng dụng trong:
    - Làm mịn biên đối tượng
    - Nối các phần rời rạc của vật thể
    - Tiền xử lý trong nhận dạng ký tự, phân đoạn ảnh

**Công thức toán học**
```math
A ⨁ B = { z | (B̂)_z ∩ A ≠ ∅ }
```
- `A`: ảnh nhị phân đầu vào
- `B`: structuring element (kernel)
- `B̂`: phép đối xứng của `B` qua gốc
- `z`: vị trí dịch kernel
- `∅`: tập rỗng

**ví dụ:**
- ảnh gốc :

            0 0 1 0 0
            0 1 1 1 0
            0 0 1 0 0

- ảnh sau khi giãn:

                    0 1 1 1 0
                    1 1 1 1 1
                    0 1 1 1 0

**Code chính:**

```python
data=Image.open('dil_img.gif').convert('L')
b=nd.binary_dilation(data, iterations=50)

c=Image.fromarray(b)
```
---

##### 2.3.2. Sử dụng binary_opening

**Mục đích:**
- Loại bỏ nhiễu nhỏ, các chi tiết sáng không mong muốn trong ảnh nhị phân
- Bảo toàn hình dạng tổng thể của vật thể lớn
- hường dùng để:
      - Làm sạch ảnh
      - Xử lý hậu cảnh
      - Tách các vật thể riêng biệt

**Công thức toán học:**
```math
A ∘ B = (A ⊖ B) ⨁ B
```
- `A`: ảnh nhị phân gốc
- `B`: structuring element (kernel)
- `⊖`: phép co (erosion)
- `⨁`: phép giãn (dilation)
- `∘`: phép mở (opening)

**Ví dụ:**
- Ảnh gốc có nhiễu:
  
        0 0 1 0 0
        0 1 1 1 0
        0 0 1 0 1
- Sau Opening:
        
        0 0 1 0 0
        0 1 1 1 0
        0 0 1 0 0

**Code chính:**
```python
a=np.array(data)
s= [[0,1,0], [1,1,1],[0,1,0]]
b= nd.binary_opening(a>0, structure=s, iterations=25)
b_uint8 = b.astype(np.uint8) * 255
c=Image.fromarray(b)
```
---

##### 2.3.3. Sử dụng binary_erosion

**Mục đích:**
- Thu nhỏ vùng sáng (pixel có giá trị 1) trong ảnh nhị phân
- Loại bỏ các chi tiết nhỏ, làm mỏng vật thể
- Ứng dụng:
      - Làm sạch ảnh khỏi các chi tiết trắng nhỏ lẻ
      - Làm mịn biên vật thể
      - Chuẩn bị cho các phép toán khác như Opening

**Công thức toán học:**
```math
A ⊖ B = { z | B_z ⊆ A }
```
- `A`: ảnh nhị phân gốc
- `B`: structuring element (kernel)
- `⊖`: phép co (erosion)
- `z`: vị trí dịch kernel
- `B_z`: kernel đặt tại vị trí `z`
- `⊆`: tập con

**ví dụ:**
- Ảnh gốc:
  
      0 1 1 1 0
      1 1 1 1 1
      0 1 1 1 0
- Sau Erosion:
  
      0 0 1 0 0
      0 1 1 1 0
      0 0 1 0 0

**Code chính:**
```python
s=[[0,1,0],[1,1,1],[0,1,0]]
b=nd.binary_erosion(data, structure=s, iterations=50)
c=Image.fromarray(b)
```
---

##### 2.3.4. Sử dụng binary_closing

**Mục đích:**
- Lấp đầy các lỗ đen nhỏ trong vật thể
- Kết nối các vùng trắng gần nhau
- Giữ nguyên hình dạng tổng thể của đối tượng
- Ứng dụng:
    - Lấp lỗ đen nhỏ trong chữ viết tay, ảnh scan
    - Làm mịn viền vật thể sáng
    - Tách các vật thể gần nhau trong nền tối

**công thức toán học:**
```math
A • B = (A ⨁ B) ⊖ B
```
- `A`: ảnh nhị phân gốc
- `B`: structuring element (kernel)
- `⨁`: phép giãn (dilation)
- `⊖`: phép co (erosion)
- `•`: phép đóng (closing)

**ví dụ:**

- Ảnh gốc có lỗ tối nhỏ:

                            1 1 1 1 1
                            1 1 0 1 1
                            1 1 1 1 1
- Sau Closing:

                            1 1 1 1 1
                            1 1 1 1 1
                            1 1 1 1 1

**code chính:**

```python
s=[[0,1,0],[1,1,1],[0,1,0]]
b=nd.binary_closing(data, structure=s, iterations=50)
c=Image.fromarray(b)
```
---

## Cấu trúc file

```
├── exercise
├── main.ipynb      
├── image.png        
├── README.md
```





  
