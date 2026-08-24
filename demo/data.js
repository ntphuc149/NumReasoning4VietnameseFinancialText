/* Ba mẫu lấy nguyên văn từ datasets/ViNumQA/origin/train.json */
/* ---------------------------------------------------------- dữ liệu ví dụ */
const EXAMPLES = [
  {
    id: "PM/2017/page_25.pdf-1",
    pre: "biểu đồ hiệu suất biểu đồ dưới đây so sánh tổng lợi nhuận tích lũy của cổ đông trên cổ phiếu phổ thông của pmi với tổng lợi nhuận tích lũy trong cùng kỳ của nhóm công ty cùng ngành của pmi và chỉ số s&p 500. biểu đồ giả định khoản đầu tư 100 đô la vào ngày 31 tháng 12 năm 2012, vào cổ phiếu phổ thông của pmi (theo giá niêm yết trên sở giao dịch chứng khoán new york) và mỗi chỉ số tại thời điểm đóng cửa thị trường và tái đầu tư cổ tức hàng quý.",
    post: "(1) nhóm công ty cùng ngành của pmi được thành lập dựa trên việc xem xét bốn đặc điểm: sự hiện diện toàn cầu; tập trung vào các sản phẩm tiêu dùng; và doanh thu thuần và vốn hóa thị trường có quy mô tương tự như của pmi. lưu ý: các số liệu được làm tròn đến 0,10 đô la gần nhất.",
    table: [
      ["ngày","pmi","nhóm công ty cùng ngành của pmi (1)","chỉ số s&p 500"],
      ["ngày 31 tháng 12 năm 2012","$ 100.00","$ 100.00","$ 100.00"],
      ["ngày 31 tháng 12 năm 2013","$ 108.50","$ 122.80","$ 132.40"],
      ["ngày 31 tháng 12 năm 2014","$ 106.20","$ 132.50","$ 150.50"],
      ["ngày 31 tháng 12 năm 2015","$ 120.40","$ 143.50","$ 152.60"],
      ["ngày 31 tháng 12 năm 2016","$ 130.80","$ 145.60","$ 170.80"],
      ["ngày 31 tháng 12 năm 2017","$ 156.80","$ 172.70","$ 208.10"]
    ],
    query: "Tỷ lệ tăng trưởng giá cổ phiếu của PMI từ năm 2012 đến 2013 là bao nhiêu?",
    trace: {
      subqueries: [
        { q:"Giá trị khoản đầu tư vào cổ phiếu PMI tại ngày 31 tháng 12 năm 2012 là bao nhiêu?", v:"$ 100.00", from:"bảng · hàng 2012, cột pmi" },
        { q:"Giá trị khoản đầu tư vào cổ phiếu PMI tại ngày 31 tháng 12 năm 2013 là bao nhiêu?", v:"$ 108.50", from:"bảng · hàng 2013, cột pmi" },
        { q:"Khoản đầu tư ban đầu được giả định trong biểu đồ là bao nhiêu?", v:"$ 100", from:"ngữ cảnh · đoạn trước bảng" }
      ],
      hits: [[2,1],[1,1]],
      plans: 15, distinct: 3, votes: [12,2,1],
      steps: [
        { ref:"#0", expr:"subtract(108.50, 100)", out:"8.50" },
        { ref:"#1", expr:"divide(#0, 100)", out:"0.085" }
      ],
      program: "subtract(108.50, 100), divide(#0, 100)",
      answer: "0.085", unit: "≈ 8,5% tăng trưởng"
    }
  },
  {
    id: "TXN/2017/page_55.pdf-2",
    pre: "tỷ lệ xu hướng chi phí chăm sóc sức khỏe giả định cho chương trình phúc lợi chăm sóc sức khỏe cho người về hưu tại hoa kỳ tính đến ngày 31 tháng 12 như sau:",
    post: "việc tăng hoặc giảm một điểm phần trăm trong tỷ lệ xu hướng chi phí chăm sóc sức khỏe trong tất cả các kỳ tương lai sẽ làm tăng hoặc giảm nghĩa vụ phúc lợi sau khi nghỉ hưu lũy kế cho chương trình tính đến ngày 31 tháng 12 năm 2017 thêm 1 triệu đô la.",
    table: [
      ["","2017","2016"],
      ["tỷ lệ xu hướng chi phí chăm sóc sức khỏe giả định cho năm tới","7.50% ( 7.50 % )","6.75% ( 6.75 % )"],
      ["tỷ lệ xu hướng cuối cùng","5.00% ( 5.00 % )","5.00% ( 5.00 % )"],
      ["năm đạt được tỷ lệ xu hướng cuối cùng","2028","2024"]
    ],
    query: "Tỷ lệ xu hướng chi phí chăm sóc sức khỏe cho năm tới đã tăng bao nhiêu điểm phần trăm trong năm 2017?",
    trace: {
      subqueries: [
        { q:"Tỷ lệ xu hướng chi phí chăm sóc sức khỏe giả định cho năm tới trong năm 2017 là bao nhiêu?", v:"7.50%", from:"bảng · hàng 1, cột 2017" },
        { q:"Tỷ lệ xu hướng chi phí chăm sóc sức khỏe giả định cho năm tới trong năm 2016 là bao nhiêu?", v:"6.75%", from:"bảng · hàng 1, cột 2016" },
        { q:"Tỷ lệ xu hướng cuối cùng của năm 2017 là bao nhiêu?", v:"5.00%", from:"bảng · hàng 2, cột 2017" }
      ],
      hits: [[1,1],[1,2]],
      plans: 15, distinct: 2, votes: [14,1],
      steps: [ { ref:"#0", expr:"subtract(7.50, 6.75)", out:"0.75" } ],
      program: "subtract(7.50, 6.75)",
      answer: "0.75", unit: "điểm phần trăm"
    }
  },
  {
    id: "HUM/2018/page_129.pdf-1",
    pre: "humana inc. thuyết minh báo cáo tài chính hợp nhất — vốn chủ sở hữu của cổ đông. bảng sau đây cung cấp chi tiết về các khoản thanh toán cổ tức, không bao gồm quyền tương đương cổ tức, trong các năm 2016, 2017 và 2018 theo chính sách cổ tức tiền mặt hàng quý đã được hội đồng quản trị phê duyệt.",
    post: "vào ngày 2 tháng 11 năm 2018, hội đồng quản trị đã công bố cổ tức bằng tiền mặt là 0,50 đô la cho mỗi cổ phiếu, được trả vào ngày 25 tháng 1 năm 2019 cho các cổ đông có tên trong sổ đăng ký vào ngày 31 tháng 12 năm 2018, với tổng số tiền là 68 triệu đô la.",
    table: [
      ["ngày thanh toán","số tiền trên mỗi cổ phiếu","tổng số tiền (tính bằng triệu)"],
      ["2016","$ 1.16","$ 172"],
      ["2017","$ 1.49","$ 216"],
      ["2018","$ 1.90","$ 262"]
    ],
    query: "Số lượng cổ phiếu đã trả cổ tức trong năm 2016 là bao nhiêu triệu cổ phiếu?",
    trace: {
      subqueries: [
        { q:"Tổng số tiền cổ tức đã thanh toán trong năm 2016 là bao nhiêu?", v:"$ 172 triệu", from:"bảng · hàng 2016, cột tổng số tiền" },
        { q:"Số tiền cổ tức trên mỗi cổ phiếu trong năm 2016 là bao nhiêu?", v:"$ 1.16", from:"bảng · hàng 2016, cột trên mỗi cổ phiếu" },
        { q:"Tổng số tiền cổ tức đã thanh toán trong năm 2017 là bao nhiêu?", v:"$ 216 triệu", from:"bảng · hàng 2017, cột tổng số tiền" }
      ],
      hits: [[1,2],[1,1]],
      plans: 15, distinct: 4, votes: [11,2,1,1],
      steps: [ { ref:"#0", expr:"divide(172, 1.16)", out:"148.27586" } ],
      program: "divide(172, 1.16)",
      answer: "148.27586", unit: "triệu cổ phiếu"
    }
  }
];
