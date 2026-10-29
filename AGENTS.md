# MASTER RULES & PROJECT WORKFLOW GUIDELINES

## PHẦN I: WORKFLOW & DEFINITION OF DONE

### 1. Definition of Done (DoD)
Một task chỉ được coi là hoàn thành khi đáp ứng đầy đủ các tiêu chuẩn kỹ thuật sau:
- **Đủ yêu cầu:** Đạt đầy đủ các yêu cầu chức năng.
- **Code chạy:** Code chạy được, thực thi không lỗi cú pháp hay runtime.
- **Test passed:** Toàn bộ kiểm thử đều vượt qua.
- **No secrets:** Không để lộ API key, token, mật khẩu hay bí mật trong code/commit.
- **Code review:** Mã nguồn đã được tự rà soát, đánh giá chất lượng.
- **GitHub:** Trạng thái mã nguồn đồng bộ với GitHub.
- **Docs updated:** Tài liệu đã được cập nhật tương ứng với các thay đổi.
- **Committed:** Thay đổi đã được commit theo chuẩn.
> **Lưu ý cốt lõi:** Viết code xong phải kiểm tra, đánh giá kỹ lưỡng thay vì chỉ thấy chạy được là hoàn thành.  
> **Implement – Plan:** Code trước, nhưng nếu mục tiêu lớn thì cần có các bước và ghi chú rõ ràng.  
> **Progress status:** Luôn cập nhật trạng thái tiến độ vào file markdown (`.md`).

### 2. Các mục bắt buộc trong báo cáo
- **Example In-out:** Ví dụ cụ thể về đầu vào và đầu ra.
- **Results:** Kết quả thực nghiệm/chạy thực tế kèm số liệu, biểu đồ hoặc minh chứng rõ ràng.
- **Limitations:** Các giới hạn còn tồn tại của giải pháp/mô hình.
- **Future Improvements:** Định hướng cải tiến trong tương lai.

### 3. AI Usage (Báo cáo sử dụng AI - file md)
Khi sử dụng AI hỗ trợ, cần ghi nhận đầy đủ:
- **Tools used:** Các công cụ AI đã sử dụng.
- **Main purposes:** Mục đích chính khi dùng AI.
- **Student contributions:** Đóng góp thực tế của sinh viên/người phát triển.
- **Important AI-generated components:** Các thành phần quan trọng do AI sinh ra.
- **Prompt purpose:** Mục đích của các prompt đã đưa ra.
- **Student modification:** Các chỉnh sửa, bổ sung do sinh viên thực hiện trên code AI sinh ra.
- **Validation:** Phương pháp kiểm chứng và đánh giá lại kết quả của AI.
- **Hội thoại & đối chiếu:** Lưu trữ tiến trình và ngữ cảnh làm việc minh bạch.

### 4. Một số lưu ý quan trọng khi dùng AI
- **Tuyệt đối không để AI đẩy code** mà không chạy thử hoặc không hiểu rõ cơ chế hoạt động.
- **Tạo test suite** để pipeline kiểm thử luôn "passed" một cách minh bạch, thực chất.
- **Bảo mật:** Không đưa API key/password lên GitHub. Không commit `.env`, token hoặc secrets vào git.
- **Tạo rule check lại** trước khi commit.
- **Tính trung thực:** Không thay đổi báo cáo tùy tiện, không sửa kết quả đánh giá thực nghiệm để làm đẹp số liệu.

### 5. Nguyên tắc viết Prompt tốt
- Cấu trúc: `Context nhỏ + Mục tiêu rõ ràng + Tiêu chí chung cụ thể`.

### 6. Lập kế hoạch (Plan)
- Xác định: `Mục tiêu + Input/Output + Feature`.
- Phân tích: `Constraints → Architecture + Modules + Milestones + Risks → Chia nhỏ viết code`.

### 7. Triển khai (Implement)
- Định rõ cho từng thành phần: `Module + Code + In/Out + Interface + Constraints`.

### 8. Sử dụng và Tài liệu (Usage và Document)
- **README.md:**
  - Project overview (Tổng quan dự án).
  - Problem statement (Bài toán cần giải quyết).
  - Main feature (Các tính năng chính).
  - System architecture (Kiến trúc hệ thống).
  - Folder structure (Cấu trúc thư mục).
  - Installation / Run / Test (Hướng dẫn cài đặt, chạy và kiểm thử).
  - Model & Knowledge Distillation (Chi tiết mô hình, chưng cất tri thức nếu áp dụng).
- **Tài liệu chi tiết khác:**
  - `Project description.md`
  - `Project design.md`

### 9. Build MVP – Triển khai Module by Module
- **Ưu tiên MVP trước:** Xây dựng sản phẩm khả dụng tối thiểu trước.
- **Quy trình module:** Nhận input → thực hiện chức năng chính → kiểm tra xem có đủ tính năng hay cần thêm tính năng phụ.
- **Một module / một lần:** Xác định rõ mục tiêu, input, output, interface trước khi sửa code.
- **Một thay đổi / một lần:** Không sửa nhiều module cùng lúc để tránh phân tán và khó kiểm soát.
- **Chu trình phát triển chuẩn:**
  `Describe → Generate → Review → Run (Smoke test) → Dry-run → Test → Refactor → Commit`

---

## PHẦN II: MASTER RULES — QUY TẮC CHUNG CHO MỌI DỰ ÁN

### 1. Nguyên tắc làm việc
- Đọc kỹ toàn bộ yêu cầu, tài liệu, đề bài, rubric và mã nguồn hiện có trước khi thực hiện.
- Xác định rõ mục tiêu, phạm vi, đầu vào, đầu ra, ràng buộc kỹ thuật và tiêu chí hoàn thành.
- Ưu tiên nội dung được cung cấp bởi người dùng, đề bài, giáo trình và tài liệu chính thức của dự án.
- Không tự ý thêm yêu cầu, thay đổi mục tiêu hoặc mở rộng phạm vi nếu không có căn cứ.
- Không bịa thông tin, kết quả, số liệu, nguồn trích dẫn, tính năng hoặc trạng thái hoàn thành.
- Phân biệt rõ thông tin từ tài liệu, thông tin đã kiểm chứng, giả định và đề xuất kỹ thuật.
- Khi thiếu thông tin quan trọng, kiểm tra những nguồn có sẵn trước; chỉ hỏi lại khi không thể đưa ra quyết định an toàn và chính xác.
- Không tuyên bố đã đọc, kiểm tra, chạy, sửa hoặc xác minh điều gì nếu thực tế chưa thực hiện.

### 2. Đọc và tuân thủ tài liệu nguồn
- Đọc đúng nội dung nguồn được cung cấp; không chỉ dựa vào tên file, mục lục, trích đoạn hoặc phần tóm tắt khi nhiệm vụ yêu cầu đọc đầy đủ.
- Giữ nguyên thuật ngữ chuyên môn, công thức, ký hiệu, định nghĩa, cấu trúc lập luận và mức độ chi tiết quan trọng của tài liệu.
- Không âm thầm thay thế nội dung nguồn bằng kiến thức bên ngoài, kể cả khi kiến thức bên ngoài có vẻ hợp lý hơn.
- Không bỏ ý, đổi nghĩa, tự hòa giải những điểm mâu thuẫn hoặc bổ sung phần tài liệu không đề cập.
- Nếu cần dùng kiến thức ngoài nguồn, phải phân biệt rõ với nội dung được trích xuất từ nguồn.
- Khi đề bài có rubric hoặc yêu cầu cụ thể, kiểm tra từng tiêu chí tương ứng trước khi kết luận đã hoàn thành.
- Với yêu cầu chính xác theo tài liệu, cần đối chiếu trực tiếp nội dung liên quan; không khẳng định chính xác hoàn toàn nếu chưa đủ căn cứ.

### 3. Quy tắc chỉnh sửa dự án có sẵn
- Trước khi sửa, kiểm tra cấu trúc thư mục, file cấu hình, mã nguồn, tài liệu, test và trạng thái hiện tại của dự án.
- Xác định vấn đề thực sự cần sửa trước khi thay đổi.
- Ưu tiên chỉnh sửa tối thiểu, có mục tiêu và có thể kiểm chứng.
- Giữ nguyên những phần đang hoạt động đúng, không liên quan đến yêu cầu.
- Không xây dựng lại toàn bộ dự án chỉ để xử lý một vấn đề cục bộ.
- Không tự ý đổi tên file, thư mục, hàm, class, biến, API, định dạng đầu vào hoặc đầu ra nếu không cần thiết.
- Không xóa dữ liệu, checkpoint, kết quả thực nghiệm, cấu hình hoặc tài liệu có giá trị.
- Trước khi sửa các thành phần có nguy cơ gây mất dữ liệu hoặc phá vỡ tính tương thích, phải xác định ảnh hưởng và có phương án bảo toàn.
- Kiểm tra các thành phần phụ thuộc trước khi sửa một thành phần dùng chung.
- Nếu dự án chưa có cấu trúc phù hợp, đề xuất hoặc xây dựng cấu trúc dựa trên yêu cầu thực tế; không áp đặt một mẫu kiến trúc cố định cho mọi dự án.

### 4. Quy tắc chất lượng mã nguồn
- Viết code rõ ràng, nhất quán, có thể đọc, bảo trì và kiểm thử.
- Tách các chức năng theo trách nhiệm hợp lý; tránh trùng lặp logic và phụ thuộc không cần thiết.
- Sử dụng tên biến, hàm, class và module thể hiện đúng mục đích.
- Xử lý các trường hợp đầu vào không hợp lệ, lỗi có thể xảy ra và tình huống biên phù hợp với phạm vi dự án.
- Không thêm abstraction, framework, thư viện hoặc tầng kiến trúc chỉ để làm dự án trông phức tạp hơn.
- Không lặp lại code, thêm file hoặc tạo cấu hình không mang lại giá trị rõ ràng.
- Không hardcode thông tin nhạy cảm, mật khẩu, API key hoặc thông tin môi trường.
- Tôn trọng phiên bản ngôn ngữ, thư viện, nền tảng và giới hạn tài nguyên hiện có.
- Giữ tương thích với giao diện và hợp đồng của những thành phần liên quan.
- Không sửa code đang đúng chỉ để tạo ra khác biệt về hình thức.

### 5. Quy tắc dữ liệu và kết quả
- Xác định nguồn gốc, cấu trúc, kiểu dữ liệu, ý nghĩa và giới hạn của dữ liệu trước khi sử dụng.
- Không tự tạo dữ liệu thực nghiệm hoặc sửa số liệu để kết quả trông tốt hơn.
- Không xóa, lọc hoặc biến đổi dữ liệu mà không có lý do phù hợp với mục tiêu.
- Kiểm tra dữ liệu thiếu, trùng lặp, sai kiểu, giá trị bất thường và các vấn đề liên quan khi chúng có ý nghĩa đối với bài toán.
- Giữ lại dữ liệu gốc khi cần thiết để có thể truy xuất và kiểm tra.
- Phân biệt dữ liệu huấn luyện, validation, test hoặc các tập dữ liệu có vai trò khác nhau khi dự án yêu cầu.
- Tránh rò rỉ dữ liệu và các phép biến đổi sử dụng thông tin từ tập đánh giá.
- Không suy diễn quan hệ nhân quả từ tương quan hoặc khẳng định kết luận vượt quá bằng chứng.
- Mọi bảng, biểu đồ và kết luận định lượng phải dựa trên dữ liệu thực tế hoặc nguồn được ghi rõ.

### 6. Quy tắc lựa chọn giải pháp kỹ thuật
- Chọn giải pháp dựa trên bài toán, tài liệu nguồn, yêu cầu, độ phức tạp, tài nguyên và khả năng kiểm chứng.
- Phân biệt yêu cầu bắt buộc với lựa chọn kỹ thuật do người triển khai đề xuất.
- Không mặc định một thuật toán, framework, mô hình hoặc kiến trúc là tốt nhất cho mọi trường hợp.
- Giải thích ngắn gọn lý do chọn giải pháp khi quyết định đó ảnh hưởng đáng kể đến thiết kế.
- Chỉ thêm tối ưu hóa khi có nhu cầu thực tế hoặc bằng chứng về vấn đề cần giải quyết.
- Cân nhắc độ chính xác, khả năng bảo trì, độ phức tạp, hiệu năng và giới hạn môi trường.
- Không thêm tính năng ngoài phạm vi chỉ vì chúng có thể hữu ích.
- Không tự thay đổi phương pháp đã được đề bài yêu cầu bắt buộc.

### 7. Kiểm thử và xác minh
- Xác định cách kiểm tra phù hợp với từng loại dự án.
- Ưu tiên chạy test hiện có trước khi sửa nếu điều kiện cho phép, để biết trạng thái ban đầu.
- Sau khi sửa, chạy các test và kiểm tra có liên quan trực tiếp đến thay đổi.
- Kiểm tra cả đầu vào hợp lệ, trường hợp biên và lỗi có thể xảy ra khi cần thiết.
- Với dự án có nhiều thành phần, kiểm tra sự tương thích giữa các thành phần bị ảnh hưởng.
- Không chỉ dựa vào việc code không báo lỗi cú pháp để kết luận chương trình hoạt động đúng.
- Không bỏ qua lỗi, warning hoặc test thất bại có ý nghĩa.
- Nếu không thể chạy test do thiếu thư viện, quyền truy cập, tài nguyên hoặc môi trường, phải nêu rõ giới hạn.
- Báo cáo chính xác test nào đã chạy, kết quả thực tế và phần nào chưa được xác minh.
- Không tuyên bố mọi thứ hoạt động hoàn hảo khi chỉ kiểm tra một phần.

### 8. Quy tắc tài liệu và văn phong
- Viết tài liệu phục vụ người đọc hiểu dự án, cách hoạt động, quyết định kỹ thuật và kết quả thực tế.
- Dùng câu trực tiếp, rõ ràng, tự nhiên và vừa đủ.
- Ưu tiên giải thích cơ chế, lý do hoặc kết quả cụ thể thay vì các tuyên bố chung chung.
- Loại bỏ câu sáo rỗng, câu lặp ý, phần tự khen dự án và những đoạn không bổ sung thông tin.
- Không lạm dụng các cụm như “đảm bảo tính tối ưu”, “tuân thủ đầy đủ”, “đáp ứng mọi yêu cầu”, “như đã đề cập ở trên” hoặc “dựa trên quy trình đã đối chiếu” khi không cần thiết.
- Không viết câu chỉ để tuyên bố rằng dự án tuân thủ một quy trình, tiêu chuẩn hoặc yêu cầu mà không giải thích thông tin thực chất.
- Không tự nhận xét mô hình, hệ thống hoặc kết quả là tốt, hiệu quả, chính xác hay tối ưu nếu chưa có bằng chứng tương ứng.
- Giữ thuật ngữ nhất quán và giải thích thuật ngữ khi đối tượng đọc cần.
- Không cố tình tạo lỗi ngữ pháp hoặc làm giảm tính chuyên nghiệp để thay đổi phong cách.
- Giữ lại những câu tổng kết, chuyển ý hoặc giải thích nếu chúng thực sự cần thiết cho mạch nội dung.
- Không thêm phần giới thiệu, kết luận, mục lục hoặc tài liệu tham khảo nếu chúng không cần thiết hoặc không được yêu cầu.

### 9. Quy tắc tổ chức dự án
- Tổ chức file theo chức năng và mức độ phức tạp thực tế.
- Phân biệt mã nguồn, dữ liệu, cấu hình, kiểm thử, tài liệu, kết quả và tài nguyên được sinh ra khi phù hợp.
- Tránh cấu trúc quá sâu, quá nhiều file nhỏ hoặc nhiều tầng thư mục không cần thiết.
- Không tạo file trùng lặp hoặc giữ nhiều phiên bản không rõ vai trò.
- Không đưa file tạm, cache, môi trường ảo, dữ liệu lớn hoặc checkpoint không cần thiết vào gói bàn giao.
- Bảo đảm các đường dẫn, cấu hình và hướng dẫn sử dụng nhất quán với cấu trúc thực tế.
- Không giả định một cấu trúc thư mục duy nhất phù hợp với mọi loại dự án.

### 10. Tài nguyên và môi trường thực thi
- Kiểm tra giới hạn phần cứng, bộ nhớ, dung lượng lưu trữ, thời gian chạy và môi trường triển khai.
- Ưu tiên giải pháp có thể thực thi trong môi trường thực tế của người dùng.
- Không mặc định có GPU, Internet, quyền quản trị hoặc khả năng cài đặt mọi thư viện.
- Hạn chế tải lại dữ liệu, tính toán hoặc cài đặt không cần thiết.
- Không xóa môi trường, dữ liệu hoặc tài nguyên đã tạo mà chưa đánh giá ảnh hưởng.
- Nếu cần thay đổi dependency, kiểm tra tính tương thích và chỉ thêm những gói thực sự cần thiết.
- Không tuyên bố đã chạy trên môi trường cụ thể nếu chưa thực sự kiểm tra.

### 11. Bảo mật và độ tin cậy
- Không để lộ thông tin xác thực, dữ liệu riêng tư hoặc thông tin nhạy cảm trong mã nguồn và tài liệu.
- Không thực hiện thao tác phá hủy dữ liệu hoặc thay đổi môi trường có phạm vi rộng mà không đánh giá rủi ro.
- Không chạy lệnh, script hoặc dependency không đáng tin cậy mà không xem xét tác động.
- Kiểm tra các điểm có thể gây lỗi bảo mật phù hợp với loại dự án.
- Không vô hiệu hóa kiểm tra, xử lý lỗi hoặc cơ chế bảo vệ chỉ để khiến test vượt qua.
- Không che giấu lỗi hoặc sửa kết quả đầu ra để tạo ấn tượng rằng hệ thống hoạt động đúng.

### 12. Quy trình thực hiện công việc
Thực hiện theo trình tự phù hợp với nhiệm vụ:
1. Tiếp nhận và phân tích yêu cầu.
2. Đọc tài liệu và kiểm tra workspace.
3. Xác định trạng thái hiện tại và các ràng buộc.
4. Xác định phần cần làm hoặc cần sửa.
5. Lựa chọn phương án phù hợp và tối thiểu.
6. Thực hiện trực tiếp trên các file liên quan.
7. Kiểm tra thay đổi và tính tương thích.
8. Chạy test hoặc xác minh trong phạm vi có thể.
9. Rà soát nội dung, cấu trúc và các vấn đề còn tồn tại.
10. Báo cáo kết quả thực tế.
*(Lưu ý: Đây là trình tự làm việc chung, không phải danh sách bắt buộc phải tạo thành các file hoặc mục tài liệu riêng trong dự án).*

### 13. Quy tắc báo cáo kết quả
Sau khi hoàn thành, báo cáo ngắn gọn nhưng đủ thông tin:
- Những gì đã thực hiện.
- Những file hoặc thành phần đã thay đổi.
- Các kiểm tra và test đã thực sự chạy.
- Kết quả xác minh.
- Những giới hạn, lỗi hoặc việc chưa hoàn thành.
- Những quyết định cần người dùng xác nhận, nếu có.
- Không liệt kê những việc chưa thực hiện như thể đã hoàn thành.
- Không tự khẳng định dự án hoàn hảo, tối ưu tuyệt đối hoặc đáp ứng mọi tiêu chí khi chưa có bằng chứng.
- Không tạo báo cáo dài dòng chỉ để mô tả lại từng thao tác nhỏ.

### 14. Quy tắc ưu tiên khi có xung đột
Khi các yêu cầu mâu thuẫn nhau, ưu tiên theo thứ tự:
1. An toàn, bảo mật và tính toàn vẹn dữ liệu.
2. Yêu cầu bắt buộc của người dùng và đề bài.
3. Tính chính xác của kiến thức, dữ liệu và kết quả.
4. Khả năng hoạt động đúng và tương thích với hệ thống hiện có.
5. Khả năng kiểm thử, bảo trì và tái sử dụng.
6. Tính đơn giản, rõ ràng và hiệu quả.
7. Sở thích về trình bày hoặc phong cách không ảnh hưởng đến các yêu cầu trên.
> Không âm thầm bỏ qua xung đột quan trọng. Nếu không thể giải quyết mà không thay đổi mục tiêu hoặc vi phạm yêu cầu bắt buộc, hãy nêu rõ vấn đề.

### 15. Chỉ dẫn cốt lõi cuối cùng
- Trước khi thực hiện bất kỳ thay đổi nào, hãy hiểu rõ trạng thái hiện tại và mục tiêu cần đạt.
- Ưu tiên tính chính xác, khả năng kiểm chứng, thay đổi tối thiểu và kết quả thực tế. Không bịa, không tự ý mở rộng phạm vi, không phá vỡ phần đang hoạt động tốt và không tạo thêm nội dung chỉ để làm dự án trông phức tạp hơn.
- Hãy thực hiện công việc trực tiếp khi đủ thông tin. Chỉ dừng lại để hỏi người dùng khi có vấn đề quan trọng không thể giải quyết an toàn hoặc chính xác bằng thông tin hiện có.
