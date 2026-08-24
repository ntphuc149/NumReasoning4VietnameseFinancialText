/* Pipeline MPR-Agent chạy thẳng trong trình duyệt.

   Cùng một hệ với `server.py`, chỉ khác chỗ chạy. Bản Python nạp `agentic/` rồi
   gọi endpoint hộ trình duyệt; bản này gọi endpoint trực tiếp bằng `fetch`.
   Khoá của người xem vốn đã nằm trong trình duyệt (chế độ BYOK), nên bỏ chặng
   trung chuyển không mất gì — mà còn bớt một nơi khoá đi qua.

   Vì sao cần: Space dạng docker đòi quota cpu-basic, tài khoản miễn phí không
   có (`limit=0`). Space tĩnh và GitHub Pages thì chỉ phục vụ file, không chạy
   Python. Đưa pipeline vào đây là cách để demo chạy thật trên cả hai chỗ đó.

   Mỗi hàm dưới đây là bản dịch của một hàm Python có sẵn; tên giữ nguyên để dò
   ngược được. Thứ tự các phần theo đúng thứ tự chạy:

     PHẦN 0  scorer.py            số, token hoá, thực thi chương trình
     PHẦN 1  agentic/llm.py       dựng ngữ cảnh C (kể cả bảng markdown)
     PHẦN 2  agentic/prompts.py   prompt Phụ lục B, nguyên văn
     PHẦN 3  agentic/program.py   phân tích kế hoạch -> chương trình -> bỏ phiếu
     PHẦN 4  transport            gọi endpoint tương thích OpenAI
     PHẦN 5  agentic/agents.py    ghép lại thành pipeline

   `tests/parity.mjs` so từng phần với bản Python trên dữ liệu thật. Sửa file
   này thì chạy lại nó trước khi đẩy đi.
*/
(function (global) {
  "use strict";

  /* ======================================================================
     PHẦN 0 — scorer.py
     ====================================================================== */

  const ALL_OPS = ["add", "subtract", "multiply", "divide", "exp", "greater",
                   "table_max", "table_min", "table_sum", "table_average"];
  const ALL_OPS_SET = new Set(ALL_OPS);
  const TABLE_OPS = new Set(["table_max", "table_min", "table_sum", "table_average"]);
  const ARITH_OPS = new Set(["add", "subtract", "multiply", "divide", "exp", "greater"]);
  const COMMUTATIVE_OPS = new Set(["add", "multiply"]);
  const JOIN = "join";
  const KNOWN_TOOLS = new Set([...ARITH_OPS, ...TABLE_OPS, JOIN]);

  const NA = "n/a";

  /* `float()` của Python, không phải `parseFloat`.

     `parseFloat("12abc")` trả 12 còn `float("12abc")` ném ValueError. Nuốt rác
     kiểu đó đúng là thứ biến một chương trình hỏng thành chương trình "hợp lệ".
     Trả `null` thay cho ném — mọi chỗ gọi đều đang bắt ValueError. */
  const _FLOAT_RE = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
  const _SPECIAL_RE = /^([+-]?)(inf(?:inity)?|nan)$/i;

  function pyFloat(text) {
    const t = String(text).trim();
    if (_FLOAT_RE.test(t)) {
      const v = parseFloat(t);
      return Number.isNaN(v) ? null : v;
    }
    const m = _SPECIAL_RE.exec(t);
    if (m) {
      if (m[2].toLowerCase() === "nan") return NaN;
      return m[1] === "-" ? -Infinity : Infinity;
    }
    return null;
  }

  function isConvertibleFloat(text) {
    return pyFloat(text) !== null;
  }

  /* `int(s)` của Python: chỉ chữ số, có dấu, không chấm không mũ. */
  const _INT_RE = /^[+-]?\d+$/;
  function isInt(text) {
    return _INT_RE.test(String(text).trim());
  }

  /* scorer.str_to_num — bộ đọc literal của FinQA, giữ nguyên. */
  function strToNum(text) {
    const raw = String(text).replace(/,/g, "");
    let v = pyFloat(raw);
    if (v !== null) return v;

    if (raw.includes("%")) {
      v = pyFloat(raw.replace(/%/g, ""));
      return v === null ? NA : v / 100.0;
    }
    if (raw.endsWith("x") || raw.endsWith("X")) {
      // Bội số viết "14.3x". Khác "%", hậu tố này không mang thang đo.
      v = pyFloat(raw.slice(0, -1));
      if (v !== null) return v;
    }
    if (raw.includes("const")) {
      let t = raw.split("const_").join("");
      if (t === "m1") t = "-1";
      v = pyFloat(t);
      return v === null ? NA : v;
    }
    return NA;
  }

  const _PAREN_NEG_RE = /^\(\s*([\d.,]+)\s*\)$/;

  function cellToNum(raw) {
    const text = String(raw).split("$").join("").trim();
    const m = _PAREN_NEG_RE.exec(text);
    if (m) {
      const value = strToNum(m[1]);
      return value === NA ? NA : -value;
    }
    return strToNum(text.split("(")[0].trim());
  }

  const _MISSING_CELL_MARKERS = new Set(
    ["", "-", "–", "—", "na", "n/a", "nan", "none"]);

  /* scorer.process_row — ô đánh dấu thiếu kỳ thì bỏ qua, ô hỏng thật huỷ hàng. */
  function processRow(rowIn) {
    const out = [];
    for (const cell of rowIn) {
      const text = String(cell).split("$").join("").trim();
      if (_MISSING_CELL_MARKERS.has(text.toLowerCase())) continue;
      const num = cellToNum(text);
      if (num === NA) return NA;
      out.push(num);
    }
    return out.length ? out : NA;
  }

  /* scorer.program_tokenization — ['op(', arg1, arg2, ')', ..., 'EOF'].

     Đếm ngoặc chứ không tách theo mọi dấu ngoặc, nên nhãn hàng kiểu `ROE (%)`
     vẫn là một token. Ném Error nếu còn chữ thừa không đọc được: chương trình
     bị cắt giữa chừng phải bị loại chứ không được tính là hợp lệ. */
  const _NAME_STICKY = /\s*([a-zA-Z_]+)\(/y;

  function programTokenization(originalProgram) {
    const text = String(originalProgram).trim();
    const program = [];
    let pos = 0;

    while (pos < text.length) {
      _NAME_STICKY.lastIndex = pos;
      const m = _NAME_STICKY.exec(text);
      if (!m) break;
      const end = _NAME_STICKY.lastIndex;
      const openIdx = end - 1;

      let depth = 0, close = -1;
      for (let i = openIdx; i < text.length; i++) {
        if (text[i] === "(") depth += 1;
        else if (text[i] === ")") {
          depth -= 1;
          if (depth === 0) { close = i; break; }
        }
      }
      if (close === -1) {
        throw new Error("Unbalanced parentheses (no matching ')' found) in program: '"
                        + originalProgram + "'");
      }

      program.push(m[1] + "(");

      const args = [];
      let argDepth = 0, current = [];
      for (const ch of text.slice(end, close)) {
        if (ch === "(") { argDepth += 1; current.push(ch); }
        else if (ch === ")") { argDepth -= 1; current.push(ch); }
        else if (ch === "," && argDepth === 0) { args.push(current.join("").trim()); current = []; }
        else current.push(ch);
      }
      if (current.length) args.push(current.join("").trim());

      for (const a of args) program.push(a);
      program.push(")");
      pos = close + 1;
      while (pos < text.length && (text[pos] === "," || text[pos] === " ")) pos += 1;
    }

    if (pos < text.length && text.slice(pos).trim()) {
      throw new Error("Trailing unparsed content in program: '" + text.slice(pos)
                      + "' (from: '" + originalProgram + "')");
    }

    program.push("EOF");
    return program;
  }

  /* scorer.extract_program — vớt chương trình ra khỏi văn bản model trả về. */
  const _NAME_SEARCH = /\s*([a-zA-Z_]+)\(/g;

  function extractProgram(rawText) {
    const text = String(rawText).replace(/```[a-zA-Z]*/g, "").split("```").join("").trim();
    const calls = [];
    let pos = 0;

    while (pos < text.length) {
      _NAME_SEARCH.lastIndex = pos;
      const m = _NAME_SEARCH.exec(text);
      if (!m) break;
      const matchEnd = m.index + m[0].length;
      const nameStart = matchEnd - 1 - m[1].length;
      if (!ALL_OPS_SET.has(m[1])) { pos = matchEnd; continue; }

      let depth = 0, close = -1;
      for (let i = matchEnd - 1; i < text.length; i++) {
        if (text[i] === "(") depth += 1;
        else if (text[i] === ")") {
          depth -= 1;
          if (depth === 0) { close = i; break; }
        }
      }
      if (close === -1) break;
      calls.push(text.slice(nameStart, close + 1).trim());
      pos = close + 1;
    }

    return calls.length ? calls.join(", ") : text;
  }

  /* scorer._steps_from_tokens — gom token thành bộ ba (op, arg1, arg2). */
  function stepsFromTokens(program) {
    const body = (program.length && program[program.length - 1] === "EOF")
      ? program.slice(0, -1) : program.slice();
    if (body.length % 4 !== 0) throw new Error("token count is not a multiple of four");
    const steps = [];
    for (let i = 0; i < body.length; i += 4) {
      const opToken = body[i], arg1 = body[i + 1], arg2 = body[i + 2], close = body[i + 3];
      if (!String(opToken).endsWith("(") || close !== ")") throw new Error("malformed step");
      const op = String(opToken).slice(0, -1).trim();
      if (!ALL_OPS_SET.has(op)) throw new Error("unknown operator '" + op + "'");
      steps.push([op, String(arg1).trim(), String(arg2).trim()]);
    }
    return steps;
  }

  /* `round(x, 5)` của Python. `toFixed` làm tròn đúng trên giá trị nhị phân
     giống `round`; chỉ phải né dạng mũ khi số quá lớn. */
  function pyRound5(x) {
    if (!Number.isFinite(x)) return x;
    if (Math.abs(x) >= 1e21) return x;
    return Number(x.toFixed(5));
  }

  /* scorer.eval_program — trả về [invalid, result]. */
  function evalProgram(program, table) {
    let thisRes = NA;
    try {
      const steps = stepsFromTokens(program);
      const resDict = {};

      for (let ind = 0; ind < steps.length; ind++) {
        const op = steps[ind][0], arg1 = steps[ind][1], arg2 = steps[ind][2];

        if (ARITH_OPS.has(op)) {
          let a, b;
          if (arg1.includes("#")) {
            const k = parseInt(arg1.split("#").join(""), 10);
            if (!(k in resDict)) throw new Error("KeyError");
            a = resDict[k];
          } else {
            a = strToNum(arg1);
            if (a === NA) return [1, NA];
          }
          if (arg2.includes("#")) {
            const k = parseInt(arg2.split("#").join(""), 10);
            if (!(k in resDict)) throw new Error("KeyError");
            b = resDict[k];
          } else {
            b = strToNum(arg2);
            if (b === NA) return [1, NA];
          }

          if (op === "add") thisRes = a + b;
          else if (op === "subtract") thisRes = a - b;
          else if (op === "multiply") thisRes = a * b;
          else if (op === "divide") {
            // Python ném ZeroDivisionError; JS trả Infinity. Ném để khớp.
            if (b === 0) throw new Error("ZeroDivisionError");
            thisRes = a / b;
          } else if (op === "exp") {
            thisRes = Math.pow(a, b);
            // Python trả số phức cho cơ số âm mũ phân số, rồi round() ném.
            if (Number.isNaN(thisRes)) throw new Error("complex result");
          } else {
            thisRes = a > b ? "yes" : "no";
          }

        } else {  // table_*
          const tableDict = {};
          for (const row of (table || [])) tableDict[String(row[0])] = row.slice(1);

          let numRow;
          if (arg1.includes("#")) {
            const k = parseInt(arg1.split("#").join(""), 10);
            if (!(k in resDict)) throw new Error("KeyError");
            numRow = [resDict[k]];
          } else {
            if (!Object.prototype.hasOwnProperty.call(tableDict, arg1)) return [1, NA];
            numRow = processRow(tableDict[arg1]);
          }
          if (numRow === NA) return [1, NA];

          if (op === "table_max") thisRes = Math.max.apply(null, numRow);
          else if (op === "table_min") thisRes = Math.min.apply(null, numRow);
          else if (op === "table_sum") thisRes = numRow.reduce((s, v) => s + v, 0);
          else thisRes = numRow.reduce((s, v) => s + v, 0) / numRow.length;
        }

        resDict[ind] = thisRes;
      }

      if (thisRes !== "yes" && thisRes !== "no" && thisRes !== NA) {
        if (typeof thisRes !== "number" || !Number.isFinite(thisRes)) {
          throw new Error("not a finite number");
        }
        thisRes = pyRound5(thisRes);
      }
    } catch (e) {
      return [1, NA];
    }
    return [0, thisRes];
  }

  /* agentic/scoring.execute_program — [ok, result]. */
  function executeProgram(program, tableRaw) {
    let tokens;
    try { tokens = programTokenization(program); }
    catch (e) { return [false, NA]; }
    const r = evalProgram(tokens, tableRaw);
    return [r[0] === 0, r[1]];
  }

  /* ======================================================================
     PHẦN 1 — agentic/llm.py: dựng ngữ cảnh C
     ====================================================================== */

  /* `format(x, "g")` của Python — 6 chữ số có nghĩa, bỏ số 0 thừa.

     Cần đúng vì `tabulate` chạy nó trên mọi ô của cột số: bảng gửi cho model
     ghi `1000`, không phải `1000.00`. Sai chỗ này là prompt khác bản Python. */
  function formatG(v, precision) {
    precision = precision || 6;
    if (Number.isNaN(v)) return "nan";
    if (!Number.isFinite(v)) return v > 0 ? "inf" : "-inf";
    if (v === 0) return Object.is(v, -0) ? "-0" : "0";

    const e = Number(v.toExponential(precision - 1).split("e")[1]);
    const strip = s => (s.indexOf(".") >= 0 ? s.replace(/0+$/, "").replace(/\.$/, "") : s);

    if (e < -4 || e >= precision) {
      const parts = v.toExponential(precision - 1).split("e");
      const mant = strip(parts[0]);
      const sign = parts[1][0] === "-" ? "-" : "+";
      let digits = parts[1].replace(/^[+-]/, "");
      if (digits.length < 2) digits = "0" + digits;
      return mant + "e" + sign + digits;
    }
    return strip(v.toFixed(Math.max(0, precision - 1 - e)));
  }

  /* tabulate: số có dấu phân nhóm nghìn ("1,234") vẫn tính là số. */
  const _THOUSANDS_NUM_RE =
    /^(?:[+-]?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]*)?|\.[0-9]+)$/;

  function isNumberCell(s) {
    const t = String(s);
    if (isConvertibleFloat(t)) {
      const v = pyFloat(t);
      // tabulate chỉ nhận inf/nan khi viết đúng ba chữ đó.
      if (!Number.isFinite(v)) return ["inf", "-inf", "nan"].indexOf(t.toLowerCase()) >= 0;
      return true;
    }
    return _THOUSANDS_NUM_RE.test(t);
  }

  /* tabulate._afterpoint — số chữ số sau dấu chấm, -1 nếu không có. */
  function afterPoint(s) {
    const t = String(s);
    if (!isNumberCell(t)) return -1;
    if (isInt(t)) return -1;
    let pos = t.lastIndexOf(".");
    if (pos < 0) pos = t.toLowerCase().lastIndexOf("e");
    return pos >= 0 ? t.length - pos - 1 : -1;
  }

  /* Kiểu của một cột: "int" < "float" < "str", lấy cái tổng quát nhất. */
  function columnType(cells) {
    let type = "int";
    for (const cell of cells) {
      const t = String(cell);
      let ct;
      if (isInt(t)) ct = "int";
      else if (isNumberCell(t)) ct = "float";
      else ct = "str";
      if (ct === "str") return "str";
      if (ct === "float") type = "float";
    }
    return type;
  }

  /* tabulate._format cho một ô. */
  function formatCell(value, type) {
    const t = String(value);
    if (type === "float") {
      // "1,234" là số với tabulate nhưng float() không đọc được: giữ nguyên văn.
      if (!isConvertibleFloat(t)) return t;
      return formatG(pyFloat(t), 6);
    }
    return t;
  }

  const MIN_PADDING = 2;

  function padLeft(width, s) { return " ".repeat(Math.max(0, width - s.length)) + s; }
  function padRight(width, s) { return s + " ".repeat(Math.max(0, width - s.length)); }

  /* `tabulate(rows, headers, tablefmt="github")`.

     Giữ nguyên cách bảng được vẽ cho baseline 0-shot/few-shot của repo. Nếu
     agent nhìn thấy bảng vẽ khác thì PA/EA của nó không còn so được với các
     dòng đã có trong README gốc. */
  function tabulateGithub(rows, headers) {
    const nCols = Math.max(headers.length, ...rows.map(r => r.length), 1);
    const head = [];
    for (let c = 0; c < nCols; c++) head.push(String(headers[c] === undefined ? "" : headers[c]));

    const cols = [];
    for (let c = 0; c < nCols; c++) {
      cols.push(rows.map(r => String(r[c] === undefined ? "" : r[c])));
    }

    const types = cols.map(columnType);
    const aligns = types.map(t => (t === "str" ? "left" : "decimal"));
    const formatted = cols.map((col, c) => col.map(v => formatCell(v, types[c])));

    // Căn theo dấu thập phân: đệm phải cho bằng số chữ số sau dấu chấm, rồi
    // căn phải cả cột.
    const aligned = formatted.map((col, c) => {
      if (aligns[c] !== "decimal") return col.slice();
      const decs = col.map(afterPoint);
      const maxDec = decs.length ? Math.max.apply(null, decs) : -1;
      return col.map((s, i) => s + " ".repeat(Math.max(0, maxDec - decs[i])));
    });

    const widths = aligned.map((col, c) => {
      let w = head[c].length + MIN_PADDING;
      for (const s of col) w = Math.max(w, s.length);
      return w;
    });

    const line = cells => "| " + cells.join(" | ") + " |";
    const out = [];
    out.push(line(head.map((h, c) =>
      aligns[c] === "left" ? padRight(widths[c], h) : padLeft(widths[c], h))));
    out.push("|" + widths.map(w => "-".repeat(w + 2)).join("|") + "|");
    for (let r = 0; r < rows.length; r++) {
      out.push(line(aligned.map((col, c) =>
        aligns[c] === "left" ? padRight(widths[c], col[r]) : padLeft(widths[c], col[r]))));
    }
    return out.join("\n");
  }

  function formatPreText(sample) { return (sample.pre_text || []).join("\n"); }
  function formatPostText(sample) { return (sample.post_text || []).join("\n"); }

  function formatTable(sample) {
    const table = sample.table || [];
    if (!table.length) return "";
    return tabulateGithub(table.slice(1), table[0]);
  }

  const _CONTEXT_FRAME_VI =
    "- Văn bản trước bảng:\n{pre_text}\n\n- Bảng:\n{table}\n\n- Văn bản sau bảng:\n{post_text}";
  const _CONTEXT_FRAME_EN =
    "- Context before table:\n{pre_text}\n\n- Table:\n{table}\n\n- Context after table:\n{post_text}";

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, (m, k) =>
      Object.prototype.hasOwnProperty.call(values, k) ? values[k] : m);
  }

  function formatContext(sample, lang) {
    const frame = lang === "vi" ? _CONTEXT_FRAME_VI : _CONTEXT_FRAME_EN;
    const empty = lang === "vi" ? "(không có)" : "(none)";
    return fill(frame, {
      pre_text: formatPreText(sample) || empty,
      table: formatTable(sample) || empty,
      post_text: formatPostText(sample) || empty,
    });
  }

  function tableRowLabels(table) {
    const out = new Set();
    for (const row of (table || [])) if (row.length > 0) out.add(String(row[0]));
    return out;
  }

  /* ======================================================================
     PHẦN 2 — agentic/prompts.py, Phụ lục B nguyên văn
     ====================================================================== */

  const VI = {
    planner_system: "Bạn là một nhà lập kế hoạch tính toán số học, có nhiệm vụ tạo một chuỗi các lời gọi hàm để giải quyết truy vấn của người dùng với khả năng song song hóa tối đa, chỉ sử dụng các công cụ được cung cấp.",

    planner_part1: `Được đưa ra một truy vấn của người dùng, hãy tạo một chuỗi các lời gọi hàm để giải quyết với khả năng song song hóa tối đa chỉ sử dụng các công cụ sau:
1. add(a: str, b: str, context: Optional[list[str]]) -> float: Cộng hai đầu vào (chuỗi số hoặc biến).
2. subtract(a: str, b: str, context: Optional[list[str]]) -> float: Trừ đầu vào thứ hai từ đầu vào thứ nhất.
3. multiply(a: str, b: str, context: Optional[list[str]]) -> float: Nhân hai đầu vào.
4. divide(a: str, b: str, context: Optional[list[str]]) -> float: Chia đầu vào thứ nhất cho đầu vào thứ hai.
5. table_max(row_identifier: str) -> float: Trả về giá trị lớn nhất trong một hàng của bảng, được xác định bằng tên hàng chính xác.
6. table_min(row_identifier: str) -> float: Trả về giá trị nhỏ nhất trong một hàng của bảng, được xác định bằng tên hàng chính xác.
7. table_sum(row_identifier: str) -> float: Trả về tổng của một hàng trong bảng, được xác định bằng tên hàng chính xác.
8. table_average(row_identifier: str) -> float: Trả về trung bình của một hàng trong bảng, được xác định bằng tên hàng chính xác.
9. join(): Kết hợp kết quả cho phản hồi cuối cùng (phải được sử dụng làm bước cuối cùng).
Hướng dẫn:
• Bạn CHỈ ĐƯỢC sử dụng các công cụ được liệt kê ở trên. Không có công cụ nào khác tồn tại hoặc có thể được sử dụng.
• PHẦN TRĂM VS TỶ LỆ: Đối với bất kỳ truy vấn nào yêu cầu "phần trăm", "thay đổi phần trăm", hoặc "tỷ lệ", kế hoạch của bạn chỉ được tính tỷ lệ cuối cùng (ví dụ, 0.25). Bạn KHÔNG ĐƯỢC nhân tỷ lệ cuối cùng với '100' để chuyển đổi thành định dạng phần trăm (ví dụ, 25%). Hàm join chỉ cần trả về tỷ lệ đã tính cuối cùng.
• YÊU CẦU BẮT BUỘC: Đối với bất kỳ truy vấn nào yêu cầu kết quả số (ví dụ, hiệu, tổng, tỷ lệ, ROI, tác động tích lũy), bạn PHẢI tạo kế hoạch với các lời gọi công cụ số (ví dụ, add, subtract, multiply, divide) để tính toán hoặc xác minh kết quả. Việc sử dụng giá trị trực tiếp từ câu trả lời truy vấn con hoặc bảng mà không có lời gọi công cụ là HOÀN TOÀN BỊ CẤM, ngay cả khi câu trả lời có vẻ rõ ràng (ví dụ, '-23' trong câu trả lời truy vấn con hoặc bảng).
• QUAN TRỌNG - Thứ tự Phép trừ cho Thay đổi:
– Đối với BẤT KỲ tính toán thay đổi nào (tăng, giảm, tăng trưởng, suy giảm, v.v.), LUÔN sử dụng: subtract(a='giá_trị_mới', b='giá_trị_cũ')
– Có nghĩa là: giá trị hiện tại/cuối cùng/mới TRỪ giá trị trước đó/ban đầu/cũ.`,

    planner_part2: `Hướng dẫn (tiếp theo):
• Truy vấn Dựa trên Bảng:
– Nếu truy vấn tham chiếu đến một bảng, trích xuất giá trị số từ bảng hoặc câu trả lời truy vấn con.
– Đối với giá trị bảng có định dạng hỗn hợp (ví dụ, '$ -54 ( 54 )'), sử dụng giá trị số (ví dụ, '-54').
– Nếu bảng cung cấp giá trị cuối cùng (ví dụ, 'tổng ảnh hưởng lũy kế' = -23), xác minh nó bằng các lời gọi công cụ.
– Sử dụng table_sum, table_max, v.v., chỉ cho các phép toán hàng.
• Đối với add, subtract, multiply, divide, đầu vào a và b PHẢI là chuỗi số (ví dụ, '3.14') hoặc biến (ví dụ, '$1') tham chiếu đến đầu ra của một hành động trước.
• Đối với các hàm bảng, row_identifier PHẢI là tên chuỗi chính xác của một hàng.
• Phân tích Đầu vào Số:
– Đảm bảo tất cả đầu vào số được phân tích cùng đơn vị.
– Đối với giá trị có dấu phần trăm (ví dụ, '48.8%'), loại bỏ dấu phần trăm và sử dụng giá trị số dưới dạng chuỗi (ví dụ, '48.8').
• Nếu "BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON" được cung cấp, hãy sử dụng các sự kiện và số liệu cụ thể từ bối cảnh đó để xây dựng kế hoạch của bạn.
• Biến ($x) CHỈ ĐƯỢC tham chiếu đến đầu ra của một hành động add, subtract, multiply, hoặc divide trước đó.
• Các biểu thức phức tạp (ví dụ, '3 * (4 + 5)') PHẢI được chia nhỏ thành các lời gọi công cụ riêng biệt.
• Kiểm tra và chuyển đổi các đơn vị để thống nhất trước khi tính toán.
• Đối với các truy vấn tài chính liên quan đến số tiền đô la (ví dụ, $47 triệu), sử dụng giá trị số dưới dạng chuỗi (ví dụ, '47').`,

    planner_part3: `Hướng dẫn (cuối cùng):
• Mỗi hành động phải có một ID duy nhất, tăng nghiêm ngặt bắt đầu từ 1.
• Hành động join PHẢI là hành động cuối cùng và KHÔNG tạo ra đầu ra số.
• KHÔNG sử dụng $id của hành động join làm biến.
• Sau join, thêm <END_OF_PLAN>.
• Tối đa hóa khả năng song song hóa bằng cách cấu trúc các phép tính độc lập để thực hiện đồng thời.
• KHÔNG phát minh hoặc gọi các công cụ không tồn tại.
Ví dụ:
Truy vấn: Thay đổi giá trị hợp lý của các công cụ thị trường tài chính từ $47 triệu năm 2009 đến $21 triệu năm 2010
Kế hoạch: 1. subtract(a='21', b='47') 2. join() <END_OF_PLAN>
Truy vấn: Phần trăm thay đổi là bao nhiêu nếu doanh thu tăng từ $500 đến $600?
Kế hoạch: 1. subtract(a='600', b='500') 2. divide(a='$1', b='500') 3. join() <END_OF_PLAN>
Truy vấn: Tỷ suất lợi nhuận tích lũy của cổ phiếu Illumina Inc. trong bốn năm kết thúc vào năm 2003 là bao nhiêu, với giá trị 100.00 vào ngày 27 tháng 7 năm 2000 và 43.81 vào ngày 26 tháng 12 năm 2003?
Kế hoạch: 1. subtract(a='43.81', b='100.00') 2. divide(a='$1', b='100.00') 3. join() <END_OF_PLAN>
Truy vấn: Tổng của 'Lợi nhuận sau thuế (tỷ đồng)' cho tất cả các năm là bao nhiêu?
Bối cảnh: [Một bảng có hàng 'Lợi nhuận sau thuế (tỷ đồng)']
Kế hoạch: 1. table_sum(row_identifier='Lợi nhuận sau thuế (tỷ đồng)') 2. join() <END_OF_PLAN>
Chỉ trả lời với danh sách nhiệm vụ theo định dạng:
idx. tool(arg_name=args)
<END_OF_PLAN>`,

    planner_query_block: "Truy vấn: {question}\nBối cảnh:\n{context}\n{subquery_block}Kế hoạch:",
  };

  const EN = {
    planner_system: "You are a numerical reasoning planner, tasked with generating a sequence of function calls to solve a user's query with maximum parallelization, using only the provided tools.",

    planner_part1: `Given a user query, generate a sequence of function calls to solve it with maximum parallelization using only the following tools:
1. add(a: str, b: str, context: Optional[list[str]]) -> float: Adds two inputs (numeric strings or variables).
2. subtract(a: str, b: str, context: Optional[list[str]]) -> float: Subtracts the second input from the first.
3. multiply(a: str, b: str, context: Optional[list[str]]) -> float: Multiplies two inputs.
4. divide(a: str, b: str, context: Optional[list[str]]) -> float: Divides the first input by the second.
5. table_max(row_identifier: str) -> float: Returns the maximum value in a table row, identified by its exact row name.
6. table_min(row_identifier: str) -> float: Returns the minimum value in a table row, identified by its exact row name.
7. table_sum(row_identifier: str) -> float: Returns the sum of a table row, identified by its exact row name.
8. table_average(row_identifier: str) -> float: Returns the average of a table row, identified by its exact row name.
9. join(): Combines results for the final response (must be used as the last step).
Instructions:
• You may ONLY use the tools listed above. No other tools exist or can be used.
• PERCENTAGE VS RATIO: For any query asking for a "percentage", "percent change", or "rate", your plan must only calculate the final ratio (e.g., 0.25). You MUST NOT multiply the final ratio by '100' to convert it to a percentage format (e.g., 25%). The join function should just return the final calculated ratio.
• MANDATORY REQUIREMENT: For any query that asks for a numerical result (e.g., difference, sum, ratio, ROI, cumulative impact), you MUST generate a plan with numerical tool calls (e.g., add, subtract, multiply, divide) to calculate or verify the result. Directly using a value from a subquery answer or table without a tool call is STRICTLY FORBIDDEN, even if the answer seems obvious (e.g., '-23' in a subquery answer or table).
• IMPORTANT - Subtraction Order for Change:
– For ANY change calculation (increase, decrease, growth, decline, etc.), ALWAYS use: subtract(a='new_value', b='old_value')
– Meaning: the current/final/new value MINUS the previous/initial/old value.`,

    planner_part2: `Instructions (continued):
• Table-Based Queries:
– If the query references a table, extract numeric values from the table or subquery answers.
– For mixed-format table values (e.g., '$ -54 ( 54 )'), use the numerical value (e.g., '-54').
– If the table provides a final value (e.g., 'total cumulative effect' = -23), verify it with tool calls.
– Use table_sum, table_max, etc., for row operations only.
• For add, subtract, multiply, divide, the inputs a and b MUST be numeric strings (e.g., '3.14') or variables (e.g., '$1') referencing the output of a prior action.
• For table functions, the row_identifier MUST be the exact string name of a row.
• Numeric Input Parsing:
– Ensure all numeric inputs are parsed in the same units.
– For values with a percentage sign (e.g., '48.8%'), strip the percent sign and use the numeric value as a string (e.g., '48.8').
• If "ADDITIONAL CONTEXT FROM SUBQUERY" is provided, use the specific facts and figures from that context to build your plan.
• Variables ($x) may ONLY refer to the output of a previous add, subtract, multiply, or divide action.
• Complex expressions (e.g., '3 * (4 + 5)') MUST be broken down into separate tool calls.
• Check and convert units for consistency before calculations.
• For financial queries involving dollar amounts (e.g., $47 million), use the numeric value as a string (e.g., '47').`,

    planner_part3: `Instructions (final):
• Each action must have a unique, strictly increasing ID starting from 1.
• The join action MUST be the final action and does NOT produce a numerical output.
• DO NOT use the $id of the join action as a variable.
• After the join, add <END_OF_PLAN>.
• Maximize parallelization by structuring independent calculations to run concurrently.
• DO NOT invent or call non-existent tools.
Examples:
Query: Change in the fair value of financial market instruments from $47 million in 2009 to $21 million in 2010
Plan: 1. subtract(a='21', b='47') 2. join() <END_OF_PLAN>
Query: What is the percentage change if revenue increases from $500 to $600?
Plan: 1. subtract(a='600', b='500') 2. divide(a='$1', b='500') 3. join() <END_OF_PLAN>
Query: What was the cumulative total return for Illumina Inc. stock for the four years ended 2003, given a value of 100.00 on July 27, 2000 and 43.81 on December 26, 2003?
Plan: 1. subtract(a='43.81', b='100.00') 2. divide(a='$1', b='100.00') 3. join() <END_OF_PLAN>
Query: What is the sum of 'Profit after tax (VND billion)' for all years?
Context: [A table with a 'Profit after tax (VND billion)' row]
Plan: 1. table_sum(row_identifier='Profit after tax (VND billion)') 2. join() <END_OF_PLAN>
Only respond with the task list in the format:
idx. tool(arg_name=args)
<END_OF_PLAN>`,

    planner_query_block: "Query: {question}\nContext:\n{context}\n{subquery_block}Plan:",
  };

  function promptSet(lang) { return lang === "vi" ? VI : EN; }

  /* Prompt trực tiếp dùng chung của repo — chỉ chạm tới khi pipeline không ra
     nổi một ứng viên nào. Câu trả lời rỗng chắc chắn 0 điểm cả hai chỉ số. */
  const SYSTEM_MESSAGE = `You are a financial analysis AI. Your task is to generate a sequential computation program to answer the question, based on the provided context.

### LIST OF 10 VALID OPERATORS:

1. add(a, b) -> a + b
2. subtract(a, b) -> a - b
3. multiply(a, b) -> a * b
4. divide(a, b) -> a / b
5. exp(a, b) -> a^b
6. greater(a, b) -> 1.0 if a > b, else 0.0
7. table_sum(row_name, none) -> sum of the numeric values in the table row named \`row_name\`
8. table_average(row_name, none) -> arithmetic mean of the numeric values in the table row named \`row_name\`
9. table_max(row_name, none) -> maximum of the numeric values in the table row named \`row_name\`
10. table_min(row_name, none) -> minimum of the numeric values in the table row named \`row_name\`

### RULES:
- Do not use free-form mathematical symbols ("+", "-", "*", "/") outside of parentheses. Every calculation must use one of the 10 operators above.
- table_* operators take exactly two arguments: the row name (copied exactly as it appears as the first cell of the target row) and the literal \`none\` (e.g. table_max(Lãi ròng, none)), never a list of numeric values.
- Do not perform mental calculations or provide explanations. The output must contain only the program string.
- Reference the result of a previous step using #0 (step 1), #1 (step 2), etc. Steps are separated by commas.
- Preserve the original number format from the context. If a value is missing, use 'none'.`;

  const USER_MESSAGE_FRAME = `### CONTEXT:
[TEXT BEFORE TABLE]
{pre_text}

[TABLE]
{table}

[TEXT AFTER TABLE]
{post_text}

### QUESTION:
{question}

### PROGRAM:`;

  /* ======================================================================
     PHẦN 3 — agentic/program.py
     ====================================================================== */

  class PlanParseError extends Error {}
  class TranspileError extends Error {}

  const _REF_RE = /^\$\s*(\d+)$/;
  const _QUOTE_PAIRS = { "'": "'", '"': '"', "‘": "’", "“": "”" };
  const _ALL_QUOTES = new Set(
    Object.keys(_QUOTE_PAIRS).concat(Object.values(_QUOTE_PAIRS)));
  const _QUOTE_CHARS = "'\"“”‘’";

  const _ARITH_ARG_SLOTS = {
    a: 0, x: 0, first: 0, value1: 0, arg1: 0, left: 0,
    b: 1, y: 1, second: 1, value2: 1, arg2: 1, right: 1,
  };
  const _TABLE_ARG_SLOTS = {
    row_identifier: 0, row: 0, row_name: 0, rowname: 0,
    name: 0, column: 0, col: 0, identifier: 0,
  };
  const _IGNORED_ARGS = new Set(["context"]);

  const _END_MARKERS = ["<END_OF_PLAN>", "<end_of_plan>"];
  // `\w` của Python là unicode; của JS chỉ ASCII. Dùng lớp unicode để một tên
  // đối số tiếng Việt (`hàng=`) vẫn được đọc là tên đối số, đúng như bản Python.
  const _ACTION_RE = /(\d+)\s*[.)\]]\s*([A-Za-z_][\p{L}\p{M}\p{N}_]*)\s*\(/u;
  const _BARE_CALL_RE = /\b([A-Za-z_][\p{L}\p{M}\p{N}_]*)\s*\(/u;
  const _IDENT_RE = /^[A-Za-z_][\p{L}\p{M}\p{N}_]*$/u;

  function refsOf(args) {
    const out = [];
    for (const arg of args) {
      const m = _REF_RE.exec(String(arg).trim());
      if (m) out.push(parseInt(m[1], 10));
    }
    return out;
  }

  function stripQuotes(text) {
    let t = String(text).trim();
    while (t.length >= 2 && _ALL_QUOTES.has(t[0]) && _ALL_QUOTES.has(t[t.length - 1])) {
      t = t.slice(1, -1).trim();
    }
    return t;
  }

  function stripChars(text, chars) {
    let s = String(text), a = 0, b = s.length;
    while (a < b && chars.indexOf(s[a]) >= 0) a++;
    while (b > a && chars.indexOf(s[b - 1]) >= 0) b--;
    return s.slice(a, b);
  }

  function splitArgs(body) {
    const args = [];
    let current = [], depth = 0, quote = null;
    for (const char of String(body)) {
      if (quote !== null) {
        current.push(char);
        if (char === quote || _QUOTE_PAIRS[quote] === char) quote = null;
        continue;
      }
      if (Object.prototype.hasOwnProperty.call(_QUOTE_PAIRS, char)) {
        quote = char; current.push(char);
      } else if (char === "(" || char === "[") { depth += 1; current.push(char); }
      else if (char === ")" || char === "]") { depth -= 1; current.push(char); }
      else if (char === "," && depth === 0) { args.push(current.join("").trim()); current = []; }
      else current.push(char);
    }
    const tail = current.join("").trim();
    if (tail) args.push(tail);
    return args;
  }

  function matchParen(text, openIdx) {
    let depth = 0, quote = null;
    for (let i = openIdx; i < text.length; i++) {
      const char = text[i];
      if (quote !== null) {
        if (char === quote || _QUOTE_PAIRS[quote] === char) quote = null;
        continue;
      }
      if (Object.prototype.hasOwnProperty.call(_QUOTE_PAIRS, char)) quote = char;
      else if (char === "(") depth += 1;
      else if (char === ")") { depth -= 1; if (depth === 0) return i; }
    }
    return -1;
  }

  function normaliseArgs(tool, rawArgs) {
    if (tool === JOIN) {
      const out = [];
      for (const a of rawArgs) {
        const idx = String(a).indexOf("=");
        const value = stripQuotes(idx >= 0 ? String(a).slice(idx + 1) : String(a));
        if (_REF_RE.test(value)) out.push(value);
      }
      return out;
    }

    const arity = TABLE_OPS.has(tool) ? 1 : 2;
    const slots = TABLE_OPS.has(tool) ? _TABLE_ARG_SLOTS : _ARITH_ARG_SLOTS;
    const filled = {};
    const positional = [];

    for (const raw of rawArgs) {
      const s = String(raw);
      const idx = s.indexOf("=");
      const name = idx >= 0 ? s.slice(0, idx) : s;
      const value = idx >= 0 ? s.slice(idx + 1) : "";
      const key = name.trim().toLowerCase();
      // Chỉ coi `x=y` là đối số có tên khi vế trái là một định danh trần;
      // `a='>=3'` và tương tự phải ở lại dạng vị trí.
      if (value && _IDENT_RE.test(key)) {
        if (_IGNORED_ARGS.has(key)) continue;
        if (Object.prototype.hasOwnProperty.call(slots, key)) {
          filled[slots[key]] = stripQuotes(value);
          continue;
        }
        positional.push(stripQuotes(value));
        continue;
      }
      positional.push(stripQuotes(s));
    }

    const ordered = [];
    let spare = 0;
    for (let slot = 0; slot < arity; slot++) {
      if (Object.prototype.hasOwnProperty.call(filled, slot)) ordered.push(filled[slot]);
      else ordered.push(spare < positional.length ? positional[spare++] : "");
    }

    if (ordered.some(p => p === "")) {
      throw new PlanParseError(tool + " expects " + arity + " argument(s), got "
                               + JSON.stringify(rawArgs));
    }
    return ordered;
  }

  function* iterCalls(text) {
    let pos = 0;
    while (pos < text.length) {
      const rest = text.slice(pos);
      const nm = _ACTION_RE.exec(rest);
      const bm = _BARE_CALL_RE.exec(rest);
      const nStart = nm ? pos + nm.index : null;
      const bStart = bm ? pos + bm.index : null;
      // Cái nào đứng trước thì thắng, hoà thì ưu tiên dạng có đánh số — regex
      // dạng trần cũng khớp tên công cụ bên trong `1. tool(`, và khớp đó bắt
      // đầu muộn hơn, nên `<=` chọn đúng cách đọc có số mà không bỏ sót lời
      // gọi không đánh số đứng trước.
      const useNumbered = nm !== null && (bm === null || nStart <= bStart);
      const m = useNumbered ? nm : bm;
      if (m === null) return;
      const start = useNumbered ? nStart : bStart;
      const tool = useNumbered ? m[2] : m[1];
      const openIdx = start + m[0].length - 1;
      const closeIdx = matchParen(text, openIdx);
      if (closeIdx === -1) return;   // sinh bị cắt: dừng ở lời gọi trọn vẹn cuối
      if (KNOWN_TOOLS.has(tool.toLowerCase())) {
        yield [useNumbered ? parseInt(m[1], 10) : null,
               tool.toLowerCase(),
               text.slice(openIdx + 1, closeIdx)];
      }
      pos = closeIdx + 1;
    }
  }

  function makePlan(actions, raw, warnings) {
    const byId = {};
    for (const a of actions) byId[a.id] = a;
    const plan = { actions, raw: raw || "", warnings: warnings || [], byId };

    plan.computation = actions.filter(a => a.tool !== JOIN);
    plan.joins = actions.filter(a => a.tool === JOIN);

    plan.answerAction = function () {
      for (let i = plan.joins.length - 1; i >= 0; i--) {
        const refs = refsOf(plan.joins[i].args);
        if (refs.length && (refs[refs.length - 1] in plan.byId)) {
          return plan.byId[refs[refs.length - 1]];
        }
      }
      if (!plan.computation.length) throw new PlanParseError("plan has no computation actions");
      return plan.computation.reduce((best, a) => (a.id > best.id ? a : best));
    };

    plan.ancestorsOf = function (action) {
      const seen = new Set();
      const stack = [action.id];
      while (stack.length) {
        const current = stack.pop();
        if (seen.has(current) || !(current in plan.byId)) continue;
        seen.add(current);
        for (const r of refsOf(plan.byId[current].args)) stack.push(r);
      }
      return seen;
    };

    plan.levels = function () {
      const depth = {};
      const resolve = (actionId, seen) => {
        if (actionId in depth) return depth[actionId];
        if (seen.has(actionId)) {
          throw new PlanParseError("cyclic dependency at action " + actionId);
        }
        const action = plan.byId[actionId];
        if (action === undefined) {
          throw new PlanParseError("dangling reference to action " + actionId);
        }
        const refs = refsOf(action.args);
        let value = 0;
        if (refs.length) {
          const next = new Set(seen); next.add(actionId);
          value = 1 + Math.max.apply(null, refs.map(r => resolve(r, next)));
        }
        depth[actionId] = value;
        return value;
      };
      for (const action of plan.computation) resolve(action.id, new Set());

      const out = [];
      const sorted = plan.computation.slice().sort(
        (x, y) => (depth[x.id] - depth[y.id]) || (x.id - y.id));
      for (const action of sorted) {
        const level = depth[action.id];
        while (out.length <= level) out.push([]);
        out[level].push(action);
      }
      return out;
    };

    plan.validate = function () {
      const known = new Set(actions.map(a => a.id));
      for (const action of actions) {
        for (const ref of refsOf(action.args)) {
          if (!known.has(ref)) {
            throw new PlanParseError("action " + action.id + " references undefined $" + ref);
          }
        }
      }
      plan.levels();
    };

    return plan;
  }

  function parsePlan(raw) {
    if (!raw || !String(raw).trim()) throw new PlanParseError("empty plan");

    let text = String(raw).replace(/```[A-Za-z]*/g, "").split("```").join("");
    for (const marker of _END_MARKERS) {
      const index = text.indexOf(marker);
      if (index !== -1) { text = text.slice(0, index); break; }
    }

    const actions = [];
    const warnings = [];
    let nextAutoId = 1;

    for (const [explicitId, tool, body] of iterCalls(text)) {
      const rawArgs = splitArgs(body);
      let args;
      try {
        args = normaliseArgs(tool, rawArgs);
      } catch (exc) {
        if (!(exc instanceof PlanParseError)) throw exc;
        if (tool === JOIN) args = [];
        else throw exc;
      }
      let actionId = explicitId !== null ? explicitId : nextAutoId;
      if (actions.some(a => a.id === actionId)) {
        warnings.push("duplicate action id " + actionId + "; renumbered");
        actionId = Math.max.apply(null, actions.map(a => a.id)) + 1;
      }
      actions.push({ id: actionId, tool, args });
      nextAutoId = Math.max(nextAutoId, actionId) + 1;
    }

    if (!actions.length) {
      throw new PlanParseError("no recognised tool calls in: "
                               + JSON.stringify(String(raw).slice(0, 200)));
    }

    const plan = makePlan(actions, raw, warnings);
    if (!plan.computation.length) throw new PlanParseError("plan contains only join()");
    plan.validate();
    return plan;
  }

  // --------------------------------------------------------- transpile ----
  const _THOUSANDS_RE = /^-?\d{1,3}(,\d{3})+(\.\d+)?%?$/;

  function cleanLiteral(raw) {
    let text = stripChars(String(raw).trim(), _QUOTE_CHARS).trim();
    text = text.split("$").join("").split(" ").join(" ").trim();
    if (_THOUSANDS_RE.test(text)) text = text.split(",").join("");
    // Tiếng Việt cũng tách nghìn bằng dấu cách: "1 234" phải thành "1234".
    // Chỉ gộp khi việc đó ra một con số, để đối số không phải số được giữ
    // nguyên và bị loại ở bước sau.
    if (text.includes(" ") && strToNum(text) === NA) {
      const squeezed = text.split(" ").join("");
      if (strToNum(squeezed) !== NA) text = squeezed;
    }
    return text;
  }

  function isRef(arg) { return _REF_RE.test(String(arg).trim()); }
  function refId(arg) {
    const m = _REF_RE.exec(String(arg).trim());
    if (!m) throw new TranspileError("not a reference: " + JSON.stringify(arg));
    return parseInt(m[1], 10);
  }

  function emitArg(arg, indexOf) {
    if (isRef(arg)) {
      const target = refId(arg);
      if (!(target in indexOf)) {
        throw new TranspileError("reference $" + target + " is not an emitted step");
      }
      return "#" + indexOf[target];
    }
    const literal = cleanLiteral(arg);
    if (!literal) throw new TranspileError("empty argument");
    if (literal.includes(",") || literal.includes("(") || literal.includes(")")) {
      throw new TranspileError("literal would break tokenisation: " + JSON.stringify(literal));
    }
    if (strToNum(literal) === NA) {
      throw new TranspileError("argument is not numeric: " + JSON.stringify(literal));
    }
    return literal;
  }

  function emitRowLabel(arg) {
    const label = stripChars(String(arg).trim(), _QUOTE_CHARS).trim();
    if (!label) throw new TranspileError("empty row label");
    if (label.includes(",")) {
      throw new TranspileError("row label contains a comma: " + JSON.stringify(label));
    }
    return label;
  }

  function topologicalOrder(plan, keep) {
    const pending = {};
    const dependents = {};
    for (const aid of keep) { dependents[aid] = []; }
    for (const aid of keep) {
      pending[aid] = refsOf(plan.byId[aid].args).filter(r => keep.has(r));
    }
    for (const aid of Object.keys(pending)) {
      for (const ref of pending[aid]) dependents[ref].push(Number(aid));
    }

    const remaining = {};
    for (const aid of keep) remaining[aid] = pending[aid].length;
    const ready = [...keep].filter(aid => remaining[aid] === 0).sort((a, b) => a - b);

    const order = [];
    while (ready.length) {
      const aid = ready.shift();
      order.push(plan.byId[aid]);
      for (const dependent of dependents[aid]) {
        remaining[dependent] -= 1;
        if (remaining[dependent] === 0) {
          ready.push(dependent);
          ready.sort((a, b) => a - b);
        }
      }
    }

    if (order.length !== keep.size) throw new TranspileError("cyclic plan");
    return order;
  }

  function transpile(plan) {
    if (!plan.computation.length) throw new TranspileError("plan has no computation actions");

    const answer = plan.answerAction();
    const compIds = new Set(plan.computation.map(a => a.id));
    const keep = new Set([...plan.ancestorsOf(answer)].filter(id => compIds.has(id)));
    if (!keep.has(answer.id)) {
      throw new TranspileError("answer action is not a computation action");
    }

    const order = topologicalOrder(plan, keep);
    if (order[order.length - 1].id !== answer.id) {
      throw new TranspileError("answer action is not last in topological order");
    }

    const indexOf = {};
    order.forEach((action, position) => { indexOf[action.id] = position; });

    const warnings = plan.warnings.slice();
    const dropped = plan.computation.length - order.length;
    if (dropped) warnings.push("pruned " + dropped + " action(s) the answer does not depend on");

    const steps = [];
    for (const action of order) {
      if (TABLE_OPS.has(action.tool)) {
        steps.push(action.tool + "(" + emitRowLabel(action.args[0]) + ", none)");
      } else if (ARITH_OPS.has(action.tool)) {
        const left = emitArg(action.args[0], indexOf);
        const right = emitArg(action.args[1], indexOf);
        steps.push(action.tool + "(" + left + ", " + right + ")");
      } else {
        throw new TranspileError("cannot emit tool " + JSON.stringify(action.tool));
      }
    }

    const program = steps.join(", ");

    // Đọc lại bằng chính bộ token hoá của bộ chấm: nó không đọc nổi thì ứng
    // viên hỏng, bất kể trông thế nào.
    try { programTokenization(program); }
    catch (exc) {
      throw new TranspileError("emitted program does not tokenise: " + exc.message);
    }

    return { program, emitted: order, warnings };
  }

  function parallelism(plan) {
    const levels = plan.levels();
    if (!levels.length) return 0.0;
    return levels.reduce((s, l) => s + l.length, 0) / levels.length;
  }

  // ------------------------------------------------------------- vote -----
  /* `format(round(v, 10), ".10g")` của Python. */
  function canonicalNumber(arg) {
    const value = strToNum(arg);
    if (value === NA) return String(arg).trim().toLowerCase();
    return formatG(Number(Number(value).toFixed(10)), 10);
  }

  function canonicalize(program) {
    const steps = stepsFromTokens(programTokenization(program));
    if (!steps.length) throw new Error("empty program");

    const render = (index, seen) => {
      if (seen.has(index)) throw new Error("cyclic reference at step " + index);
      const step = steps[index];
      if (step === undefined) throw new Error("missing step " + index);
      const op = step[0], arg1 = step[1], arg2 = step[2];
      if (TABLE_OPS.has(op)) return op + "(" + arg1.trim().toLowerCase() + ")";
      const parts = [];
      for (const arg of [arg1, arg2]) {
        const text = arg.trim();
        if (text.startsWith("#")) {
          const next = new Set(seen); next.add(index);
          parts.push(render(parseInt(text.slice(1), 10), next));
        } else {
          parts.push(canonicalNumber(text));
        }
      }
      if (COMMUTATIVE_OPS.has(op)) parts.sort();
      return op + "(" + parts[0] + "," + parts[1] + ")";
    };

    // Gốc là bước cuối, nên nhánh chết không tách cụm.
    return render(steps.length - 1, new Set());
  }

  function nSteps(program) {
    try { return stepsFromTokens(programTokenization(program)).length; }
    catch (e) { return 1000000; }   // không đọc được: không bao giờ thắng hoà
  }

  function clusterCanonical(programs) {
    const byKey = new Map();
    programs.forEach((program, index) => {
      let key;
      try { key = canonicalize(program); }
      catch (e) { key = "__unparsed__::" + String(program).trim(); }
      let cluster = byKey.get(key);
      if (cluster === undefined) {
        cluster = { key, members: [], representative: program };
        byKey.set(key, cluster);
      }
      cluster.members.push(index);
    });
    return [...byKey.values()];
  }

  /* `p* = argmax Score(p_i)` — phương trình (4).

     Score là tần suất cụm. Hoà thì ít bước hơn thắng, rồi tới chỉ số sinh sớm
     nhất: n mẫu độc lập không có thứ tự nào có nghĩa, nên "sinh trước" là mốc
     ổn định duy nhất, và cũng là quy ước bỏ phiếu majority-of-k của repo. */
  function vote(programs) {
    if (!programs.length) return { winner: null, winnerIndex: null, clusters: [], consensus: 0 };

    const clusters = clusterCanonical(programs);
    const bestOf = cluster => cluster.members.slice().sort(
      (i, j) => (nSteps(programs[i]) - nSteps(programs[j])) || (i - j))[0];

    for (const cluster of clusters) cluster.representative = programs[bestOf(cluster)];

    const rank = cluster => {
      const best = bestOf(cluster);
      return [-cluster.members.length, nSteps(programs[best]), best];
    };
    const cmp = (a, b) => {
      const ra = rank(a), rb = rank(b);
      for (let i = 0; i < ra.length; i++) if (ra[i] !== rb[i]) return ra[i] - rb[i];
      return 0;
    };

    const winnerCluster = clusters.slice().sort(cmp)[0];
    const winnerIndex = bestOf(winnerCluster);

    const total = clusters.reduce((s, c) => s + c.members.length, 0);
    const best = clusters.reduce((a, c) => (c.members.length > a.members.length ? c : a));
    const consensus = total ? best.members.length / total : 0;

    return {
      winner: programs[winnerIndex],
      winnerIndex,
      clusters: clusters.slice().sort((a, b) => b.members.length - a.members.length),
      consensus,
    };
  }

  /* ======================================================================
     PHẦN 4 — transport: endpoint tương thích OpenAI qua `fetch`
     ====================================================================== */

  const _REASONING_KEYWORDS = ["r1", "thinking", "reasoner", "qwq", "o1", "o3", "reasoning"];
  const KNOWN_REASONING_MODELS = new Set(
    ["DeepSeek-V4-Flash", "GLM-5.2", "gpt-oss-120b", "gpt-oss-20b"]);
  const HIDDEN_REASONING_FIELDS = ["reasoning_content", "reasoning"];
  const REASONING_MODEL_MIN_TOKENS = 2048;

  function isReasoningModel(name) {
    if (KNOWN_REASONING_MODELS.has(name)) return true;
    const n = String(name).toLowerCase();
    return _REASONING_KEYWORDS.some(kw => n.includes(kw));
  }

  function hiddenReasoning(message) {
    if (!message) return "";
    for (const field of HIDDEN_REASONING_FIELDS) {
      const value = message[field];
      if (value) return String(value);
    }
    return "";
  }

  class LLMError extends Error {}

  const sleep = ms => new Promise(r => setTimeout(r, ms));

  class LLMClient {
    constructor(config, apiKey, baseUrl) {
      this.config = config;
      this.apiKey = apiKey;
      this.baseUrl = String(baseUrl).replace(/\/+$/, "");
      this.usage = { requests: 0, prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
      this._serverSideN = config.use_server_side_n === undefined ? null : config.use_server_side_n;
      this._sendTopK = config.send_top_k !== false;
      this._probe = null;
      this.reasoning = [];
    }

    _url() { return this.baseUrl + "/chat/completions"; }

    _body(messages, model, maxTokens, temperature, n) {
      const cfg = this.config;
      const body = {
        model,
        messages,
        max_tokens: maxTokens,
        temperature: temperature === null || temperature === undefined
          ? cfg.temperature : temperature,
        top_p: cfg.top_p,
        stream: false,
      };
      if (this._sendTopK) body.top_k = cfg.top_k;
      if (n !== null && n !== undefined && n > 1) body.n = n;
      return body;
    }

    async _post(body, signal) {
      const res = await fetch(this._url(), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: "Bearer " + this.apiKey,
        },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        let detail = "";
        try { detail = await res.text(); } catch (e) { /* body đã đóng */ }
        const err = new Error("HTTP " + res.status + " " + detail.slice(0, 400));
        err.status = res.status;
        err.detail = detail;
        throw err;
      }
      return res.json();
    }

    _recordUsage(data) {
      const u = data && data.usage;
      this.usage.requests += 1;
      if (!u) return 0;
      const prompt = u.prompt_tokens || 0;
      const completion = u.completion_tokens || 0;
      const total = u.total_tokens || (prompt + completion);
      this.usage.prompt_tokens += prompt;
      this.usage.completion_tokens += completion;
      this.usage.total_tokens += total;
      return total;
    }

    /* Bản dịch của `LLMClient._call`, giữ cả hai bài học đã trả giá:
       bỏ `top_k`/`n` khi endpoint từ chối, và nhân đôi ngân sách token khi
       model reasoning tiêu hết budget vào chuỗi suy nghĩ mà chưa ra câu trả
       lời (content rỗng nhưng có vết suy nghĩ). */
    async _call(messages, model, maxTokens, temperature, n, opts) {
      const cfg = this.config;
      opts = opts || {};
      let effectiveMaxTokens = maxTokens;
      if (isReasoningModel(model) && effectiveMaxTokens < REASONING_MODEL_MIN_TOKENS) {
        effectiveMaxTokens = REASONING_MODEL_MIN_TOKENS;
      }
      let escalations = 0;
      const MAX_ESCALATIONS = 2;
      let lastError = null;
      const truncated = [];
      let nn = n;

      for (let attempt = 1; attempt <= cfg.max_retries; attempt++) {
        let data;
        try {
          data = await this._post(
            this._body(messages, model, effectiveMaxTokens, temperature, nn), opts.signal);
        } catch (exc) {
          if (exc && exc.name === "AbortError") throw exc;
          lastError = exc;
          const text = String((exc && exc.detail) || (exc && exc.message) || exc);
          // Endpoint từ chối top_k (hoặc n) một cách xác định; gửi lại y hệt
          // mãi mãi là vô ích. Bỏ tham số rồi thử lại ngay.
          if (this._sendTopK && text.includes("top_k")) { this._sendTopK = false; continue; }
          if (nn && nn > 1 && text.toLowerCase().includes("'n'")) {
            this._serverSideN = false; nn = null; continue;
          }
          if (exc && exc.status === 429) { await sleep(65000); continue; }
          if (attempt < cfg.max_retries) await sleep(cfg.retry_base_delay * 1000 * attempt);
          continue;
        }

        this._recordUsage(data);

        const outputs = [];
        let starved = false;
        for (const choice of (data.choices || [])) {
          const message = choice.message || {};
          const content = (message.content || "").trim();
          const trace = hiddenReasoning(message);
          if (trace && effectiveMaxTokens > 16) this.reasoning.push(trace.trim());
          if (!content) {
            // Phân biệt "bị cắt giữa chuỗi suy nghĩ" (đáng tăng ngân sách) với
            // "model trả về rỗng" (tăng cũng vô ích): cái trước để lại vết.
            if (trace) starved = true;
            continue;
          }
          if (choice.finish_reason === "length") truncated.push(content);
          else outputs.push(content);
        }
        if (outputs.length) return outputs;

        if (starved && escalations < MAX_ESCALATIONS) {
          escalations += 1;
          effectiveMaxTokens *= 2;
          continue;
        }
        if (attempt < cfg.max_retries) await sleep(cfg.retry_base_delay * 1000 * attempt);
      }

      if (truncated.length) return truncated;
      throw new LLMError(model + ": tất cả " + cfg.max_retries + " lần thử đều thất bại"
                         + (lastError ? " (" + String(lastError.message || lastError).slice(0, 200) + ")" : ""));
    }

    async complete(system, user, model, maxTokens, temperature, opts) {
      const messages = [];
      if (system) messages.push({ role: "system", content: system });
      messages.push({ role: "user", content: user });
      const out = await this._call(messages, model, maxTokens, temperature, null, opts);
      return out[0];
    }

    /* Thăm dò một lần bằng một yêu cầu n=2 thật, rồi nhớ kết quả.

       Rẻ hơn đoán: endpoint có tôn trọng `n` thì một kế hoạch n=15 chỉ tốn một
       lần tính tiền prompt thay vì mười lăm. */
    async supportsServerSideN(model, opts) {
      if (this._serverSideN !== null) return this._serverSideN;
      if (this._probe) return this._probe;
      this._probe = (async () => {
        try {
          const data = await this._post({
            model, messages: [{ role: "user", content: "Say OK." }],
            max_tokens: 8, temperature: 1.0, n: 2,
          }, opts && opts.signal);
          this._recordUsage(data);
          this._serverSideN = (data.choices || []).length >= 2;
        } catch (e) {
          this._serverSideN = false;
        }
        return this._serverSideN;
      })();
      return this._probe;
    }

    /* `P_n-sample` — phương trình (3): n lần sinh độc lập.

       Dùng một yêu cầu `n=n` phía máy chủ khi endpoint hỗ trợ, và bù thêm nếu
       trả về ít hơn số xin. Không hỗ trợ thì bắn n yêu cầu đơn song song. */
    async sampleN(system, user, model, n, maxTokens, temperature, opts) {
      opts = opts || {};
      const onProgress = opts.onProgress || (() => {});
      if (n <= 1) {
        const one = await this.complete(system, user, model, maxTokens, temperature, opts);
        onProgress(1);
        return [one];
      }

      const messages = [];
      if (system) messages.push({ role: "system", content: system });
      messages.push({ role: "user", content: user });

      let outputs = [];
      if (await this.supportsServerSideN(model, opts)) {
        try {
          outputs = await this._call(messages, model, maxTokens, temperature, n, opts);
        } catch (e) {
          if (e && e.name === "AbortError") throw e;
          outputs = [];
        }
        onProgress(Math.min(outputs.length, n));
        if (outputs.length >= n) return outputs.slice(0, n);
      }

      const remaining = n - outputs.length;
      if (remaining > 0) {
        let done = outputs.length;
        const jobs = [];
        for (let i = 0; i < remaining; i++) {
          jobs.push((async () => {
            try {
              const r = await this._call(messages, model, maxTokens, temperature, null, opts);
              return r[0];
            } catch (e) {
              if (e && e.name === "AbortError") throw e;
              return null;
            } finally {
              done += 1;
              onProgress(Math.min(done, n));
            }
          })());
        }
        for (const r of await Promise.all(jobs)) if (r) outputs.push(r);
      }

      if (!outputs.length) throw new LLMError(model + ": n-sampling không ra kết quả dùng được");
      return outputs;
    }
  }

  /* ======================================================================
     PHẦN 5 — agentic/agents.py: ghép thành pipeline
     ====================================================================== */

  /* Mặc định của paper. Bản demo bỏ hai node phân rã (§4.1/§4.2) — đúng dòng
     "Multi-path only" trong bảng ablation. Theo số của paper, bỏ phần này chỉ
     mất ~0,1-0,4 EA. */
  function makeConfig(model, nSamples) {
    return {
      model_planner: model,
      model_fallback: model,
      n_samples: nSamples,          // paper: 15
      temperature: 0.6,             // paper: 0.6
      top_p: 0.95,                  // paper: 0.95
      top_k: 20,                    // paper: 20
      send_top_k: true,
      max_tokens_planner: 768,
      max_tokens_fallback: 512,
      prompt_lang: "vi",            // phụ lục B.1/B.3/B.5... là tiếng Việt
      vote_mode: "canonical",       // §4.4
      use_decomposition: false,
      validate_row_labels: true,
      drop_invalid_candidates: true,
      use_direct_prompt_fallback: true,
      max_retries: 5,
      retry_base_delay: 2.0,
      use_server_side_n: undefined,
    };
  }

  function buildState(sample, lang) {
    return {
      question: (sample.qa && sample.qa.question) || "",
      context: formatContext(sample, lang),
      pre_text: formatPreText(sample),
      table: formatTable(sample),
      post_text: formatPostText(sample),
      table_raw: sample.table,
    };
  }

  /* Planner.build_user_prompt — B.7 + B.9 + B.11 rồi khối mang câu hỏi.

     Ba phần hướng dẫn được in thành ba hình riêng trong paper nhưng được mô tả
     ở đó là một "comprehensive, multi-part user prompt", nên nối theo thứ tự. */
  function buildPlannerPrompt(state, cfg) {
    const P = promptSet(cfg.prompt_lang);
    const parts = [P.planner_part1, P.planner_part2, P.planner_part3];
    parts.push(fill(P.planner_query_block, {
      question: state.question,
      context: state.context,
      subquery_block: "",   // phân rã đã tắt
    }));
    return parts.join("\n\n");
  }

  function unknownRowLabels(program, tableRaw) {
    let steps;
    try { steps = stepsFromTokens(programTokenization(program)); }
    catch (e) { return []; }
    const known = tableRowLabels(tableRaw);
    const missing = new Set();
    for (const [op, arg1] of steps) {
      if (TABLE_OPS.has(op) && !arg1.startsWith("#") && !known.has(arg1)) missing.add(arg1);
    }
    return [...missing].sort();
  }

  function buildCandidate(index, rawPlan, tableRaw, cfg) {
    const candidate = {
      index, raw_plan: rawPlan, plan: null, program: null,
      parallelism: 0, warnings: [], executable: false,
      exe_result: null, error: "",
    };

    let plan;
    try { plan = parsePlan(rawPlan); }
    catch (exc) { candidate.error = "parse: " + exc.message; return candidate; }
    candidate.plan = plan;
    try { candidate.parallelism = parallelism(plan); } catch (e) { candidate.parallelism = 0; }

    let result;
    try { result = transpile(plan); }
    catch (exc) { candidate.error = "transpile: " + exc.message; return candidate; }
    candidate.program = result.program;
    candidate.warnings = result.warnings;

    if (cfg.validate_row_labels) {
      const missing = unknownRowLabels(result.program, tableRaw);
      if (missing.length) {
        candidate.error = "unknown table row(s): " + JSON.stringify(missing);
        return candidate;
      }
    }

    const [ok, value] = executeProgram(result.program, tableRaw);
    candidate.executable = ok;
    candidate.exe_result = value;
    if (!ok) candidate.error = "execution returned n/a";
    return candidate;
  }

  // ---------------------------------------------- phụ trợ cho giao diện ----
  const NUM_RE = /-?\d[\d.,]*\d|-?\d/g;
  const REF_RE = /#\d+/g;

  /* server.to_float — `100,00`, `1.234,56`, `1,234.56` -> số. */
  function toFloat(token) {
    let t = String(token).trim().replace(/[.,]+$/, "");
    if (!t) return null;
    const hasDot = t.includes("."), hasComma = t.includes(",");
    if (hasDot && hasComma) {
      const dec = t.lastIndexOf(".") > t.lastIndexOf(",") ? "." : ",";
      t = t.split(dec === "." ? "," : ".").join("").split(dec).join(".");
    } else if (hasComma) {
      const tail = t.slice(t.lastIndexOf(",") + 1);
      t = tail.length === 3 ? t.split(",").join("") : t.split(",").join(".");
    } else if (hasDot) {
      const parts = t.split(".");
      const tail = parts[parts.length - 1];
      if (tail.length === 3 && parts[0].length <= 3 && parts.length > 2) t = parts.join("");
    }
    const v = Number(t);
    return Number.isNaN(v) ? null : v;
  }

  function cellsMatching(wanted, table) {
    const hits = [];
    for (let r = 1; r < (table || []).length; r++) {
      for (let c = 1; c < table[r].length; c++) {
        const found = String(table[r][c]).match(NUM_RE) || [];
        const vals = found.map(toFloat).filter(v => v !== null);
        if (vals.some(cv => wanted.some(w => Math.abs(cv - w) < 1e-9))) hits.push([r, c]);
      }
    }
    return hits.slice(0, 8);
  }

  function programHits(program, table) {
    const body = String(program || "").replace(REF_RE, " ");
    const found = body.match(NUM_RE) || [];
    const wanted = found.map(toFloat).filter(v => v !== null);
    return cellsMatching(wanted, table);
  }

  /* Chạy từng tiền tố của chương trình để lấy giá trị trung gian #0, #1… */
  function splitSteps(program, tableRaw) {
    const parts = String(program).split(/,\s*(?=[a-z_]+\()/).map(p => p.trim()).filter(Boolean);
    const steps = [];
    for (let i = 0; i < parts.length; i++) {
      const [ok, value] = executeProgram(parts.slice(0, i + 1).join(", "), tableRaw);
      steps.push({ ref: "#" + i, expr: parts[i], out: ok ? String(value) : "n/a" });
    }
    return steps;
  }

  function toSample(payload) {
    const lines = text => String(text || "").split("\n").map(p => p.trim()).filter(Boolean);
    return {
      id: "demo",
      pre_text: lines(payload.pre),
      post_text: lines(payload.post),
      table: (payload.table || []).map(row => row.map(c => String(c))),
      qa: { question: String(payload.query || "").trim() },
    };
  }

  const STAGES = {
    planner: ["Lập kế hoạch", ""],
    equation_extractor: ["Bỏ phiếu & thực thi", ""],
  };

  /* Cùng chuỗi sự kiện mà `server.py` bắn qua SSE, để giao diện không phải
     biết pipeline đang chạy ở đâu. */
  async function runPipeline(payload, emit, opts) {
    opts = opts || {};
    const maxN = opts.maxN || 15;
    let nSamples = parseInt(payload.n_samples, 10) || 15;
    nSamples = Math.max(1, Math.min(nSamples, maxN));

    const apiKey = String(payload.api_key || "").trim();
    const baseUrl = String(payload.base_url || "").trim();
    if (!apiKey || !baseUrl) {
      throw new Error("Chưa có khoá API. Bấm “Khoá API” trên đầu trang để nhập.");
    }
    if (!/^https:\/\//i.test(baseUrl)) throw new Error("BASE_URL phải bắt đầu bằng https://");

    const model = String(payload.model || "").trim();
    if (!model) throw new Error("Chưa chọn model.");

    const cfg = makeConfig(model, nSamples);
    const client = new LLMClient(cfg, apiKey, baseUrl);
    const sample = toSample(payload);
    const state = buildState(sample, cfg.prompt_lang);

    const visible = ["planner", "equation_extractor"];
    emit({
      type: "run_start", model, n_samples: nSamples,
      vote_mode: cfg.vote_mode, prompt_lang: cfg.prompt_lang,
      decomposition: cfg.use_decomposition,
      stages: visible.map(k => ({ key: k, name: STAGES[k][0], sub: STAGES[k][1] })),
    });

    // ---- node 3: Planner -------------------------------------------------
    emit({ type: "stage", i: 0, key: "planner", status: "start" });
    let t0 = performance.now();
    const rawPlans = await client.sampleN(
      promptSet(cfg.prompt_lang).planner_system,
      buildPlannerPrompt(state, cfg),
      cfg.model_planner, nSamples, cfg.max_tokens_planner, null,
      { signal: opts.signal, onProgress: k => emit({ type: "plans", sampled: k }) });
    let seconds = Math.round((performance.now() - t0) / 10) / 100;

    emit({ type: "plans", sampled: rawPlans.length });
    if (client.reasoning.length) {
      // Ghép được với từng kế hoạch chỉ khi endpoint trả n lựa chọn trong một
      // lần gọi — lúc đó thứ tự khớp. Phải bắn n lệnh song song thì thứ tự về
      // là thứ tự xong, không ghép bừa.
      const paired = client._serverSideN === true && client.reasoning.length === rawPlans.length;
      emit({ type: "reasoning", items: client.reasoning.slice(0, nSamples), paired });
    }
    emit({
      type: "stage", i: 0, key: "planner", status: "done", seconds,
      note: "n = " + nSamples + ", T = " + cfg.temperature,
    });

    // ---- node 4: Equation Extractor -------------------------------------
    emit({ type: "stage", i: 1, key: "equation_extractor", status: "start" });
    t0 = performance.now();

    const candidates = rawPlans.map((raw, i) => buildCandidate(i, raw, state.table_raw, cfg));
    const pool = cfg.drop_invalid_candidates
      ? candidates.filter(c => c.program && c.executable)
      : candidates.filter(c => c.program !== null);

    let program = "", answer = null, fallback = null, voteResult = null;
    const errors = [];

    if (pool.length) {
      voteResult = vote(pool.map(c => c.program));
      if (voteResult.winner !== null) {
        program = voteResult.winner;
        const [ok, value] = executeProgram(program, state.table_raw);
        answer = ok ? value : null;
      }
    }

    if (!program && cfg.use_direct_prompt_fallback) {
      // Không ứng viên nào dùng được: hạ xuống prompt trực tiếp của repo thay
      // vì trả chuỗi rỗng — rỗng thì chắc chắn 0 điểm cả hai chỉ số.
      try {
        const raw = await client.complete(
          SYSTEM_MESSAGE,
          fill(USER_MESSAGE_FRAME, {
            pre_text: state.pre_text, table: state.table,
            post_text: state.post_text, question: state.question,
          }),
          cfg.model_fallback, cfg.max_tokens_fallback, 0.0, { signal: opts.signal });
        let candidate = extractProgram(raw);
        try { programTokenization(candidate); } catch (e) { candidate = ""; }
        fallback = "direct_prompt";
        program = candidate;
        const [ok, value] = executeProgram(program, state.table_raw);
        answer = ok ? value : null;
      } catch (exc) {
        if (exc && exc.name === "AbortError") throw exc;
        errors.push("equation_extractor: fallback failed: " + exc.message);
      }
    }

    seconds = Math.round((performance.now() - t0) / 10) / 100;
    const clusters = voteResult
      ? voteResult.clusters.map(c => c.members.length).sort((a, b) => b - a) : [];

    emit({ type: "hits", cells: programHits(program, sample.table) });
    emit({
      type: "candidates",
      items: candidates.map(c => ({
        i: c.index,
        plan: (c.raw_plan || "").trim(),
        program: c.program || "",
        ok: Boolean(c.program && c.executable),
        result: c.exe_result === null ? "" : String(c.exe_result),
        error: c.error || "",
      })),
      winner: program,
    });
    emit({
      type: "final",
      program,
      answer: answer === null ? "" : String(answer),
      steps: program ? splitSteps(program, state.table_raw) : [],
      votes: clusters.length ? clusters : [pool.length],
      distinct: clusters.length,
      plans: rawPlans.length,
      usable: pool.length,
      consensus: voteResult ? Math.round(voteResult.consensus * 1000) / 1000 : 0,
      fallback,
      seconds,
    });
    emit({
      type: "stage", i: 1, key: "equation_extractor", status: "done", seconds,
      note: fallback ? "dự phòng" : "đồng thuận",
    });

    emit({ type: "run_end", errors, usage: client.usage });
  }

  // ------------------------------------------------------------ exports ---
  global.MPR = {
    runPipeline,
    // Được kiểm bởi tests/parity.mjs — giữ nguyên tên khi sửa.
    strToNum, cellToNum, processRow, programTokenization, extractProgram,
    stepsFromTokens, evalProgram, executeProgram,
    formatG, tabulateGithub, formatTable, formatContext,
    parsePlan, transpile, canonicalize, vote, nSteps, parallelism, cleanLiteral,
    buildPlannerPrompt, buildState, toSample, makeConfig,
    splitSteps, programHits, toFloat,
    LLMClient, LLMError, PlanParseError, TranspileError,
    VI, EN, SYSTEM_MESSAGE, USER_MESSAGE_FRAME,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = global.MPR;

})(typeof globalThis !== "undefined" ? globalThis : this);
