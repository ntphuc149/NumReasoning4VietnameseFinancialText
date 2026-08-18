"""Every prompt the pipeline sends, in one place.

Three groups, kept apart on purpose:

1. **Appendix B, verbatim.** `VI` and `EN` below carry the paper's own prompts,
   transcribed exactly -- including the paper's typo ("bao nhiều" for "bao
   nhiêu"). Prompt fidelity is the point; nothing here is paraphrased,
   reordered, or "improved".

       VI: B.1, B.3, B.5, B.7, B.9, B.11   (Vietnamese)
       EN: B.2, B.4, B.6, B.8, B.10, B.12  (the paper's own English renderings)

   One field is *not* verbatim and is marked as such: `planner_query_block`. The
   paper prints B.7/B.9/B.11 as the static instruction text but never shows the
   block carrying the actual question, context, and subquery answers. It is
   reconstructed from two things the paper does show -- B.9's bullet referring
   to a section headed "BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON", and the
   `Truy vấn:` / `Bối cảnh:` / `Kế hoạch:` shape of B.11's worked examples.

2. **Opt-in extensions**, NOT from the paper, off unless `use_prompt_ext`.

3. **The repo's shared direct prompt**, used only as a last-resort fallback.
"""

from dataclasses import dataclass

# =============================================================================
# 1. APPENDIX B -- VERBATIM
# =============================================================================


@dataclass(frozen=True)
class PromptSet:
    """One language's full set of Appendix B prompts."""

    subquery_generator: str      # B.1 / B.2
    subquery_answerer: str       # B.3 / B.4
    planner_system: str          # B.5 / B.6
    planner_part1: str           # B.7 / B.8   tools + core instructions
    planner_part2: str           # B.9 / B.10  detailed instructions
    planner_part3: str           # B.11 / B.12 final rules + examples
    planner_query_block: str     # reconstructed -- see module docstring
    subquery_block_header: str   # reconstructed -- see module docstring


# ------------------------------------------------------------- Vietnamese ---
VI = PromptSet(
    # ---------------------------------------------------- Figure B.1 (G_sq) --
    subquery_generator="""Bạn là chuyên gia trong việc chia nhỏ câu hỏi của người dùng thành một chuỗi gồm 3 đến 5 câu hỏi con nhỏ hơn, dùng để trích xuất dữ liệu. Được đưa ra câu hỏi của người dùng và bối cảnh xung quanh (văn bản và bảng), nhiệm vụ của bạn là xác định các điểm dữ liệu thô cần thiết để trả lời câu hỏi chính.
Mục tiêu của bạn: Tạo danh sách các câu hỏi chỉ TRÍCH XUẤT giá trị dữ liệu thô hoặc văn bản hoặc phạm vi dữ liệu thô.
HƯỚNG DẪN QUAN TRỌNG:
• KHÔNG tạo ra các câu hỏi thực hiện so sánh (ví dụ, 'cái nào thấp nhất/cao nhất').
• KHÔNG tạo ra các câu hỏi thực hiện tính toán (ví dụ, 'tổng/trung bình/hiệu là bao nhiều').
• KHÔNG tạo ra các câu hỏi yêu cầu câu trả lời cuối cùng.
• Công việc của bạn chỉ là hỏi về chính các điểm dữ liệu. Bước tiếp theo trong quy trình sẽ thực hiện tính toán thực tế.
Bây giờ, thực hiện nhiệm vụ này cho đầu vào sau đây.
Bối cảnh:
{context}
Câu hỏi gốc: {question}
Trả về các câu hỏi con dưới dạng đối tượng JSON với một khóa duy nhất "subqueries" chứa danh sách các chuỗi.""",
    # ---------------------------------------------------- Figure B.3 (A_sq) --
    subquery_answerer="""Bạn là trợ lý hữu ích. Nhiệm vụ của bạn là trả lời truy vấn con được đưa ra chỉ dựa trên bối cảnh được cung cấp.
Bối cảnh bao gồm văn bản trước bảng, chính bảng đó, và văn bản sau bảng. Luôn phản hồi bằng các câu hoàn chỉnh, nêu rõ câu trả lời cho câu hỏi hoặc tính toán.
Cung cấp câu trả lời ngắn gọn và trực tiếp cho truy vấn con.
Bối cảnh:
{context}
Truy vấn con:
{subquery}
Trả lời:""",
    # ------------------------------------------------------------ Figure B.5 --
    planner_system="""Bạn là một nhà lập kế hoạch tính toán số học, có nhiệm vụ tạo một chuỗi các lời gọi hàm để giải quyết truy vấn của người dùng với khả năng song song hóa tối đa, chỉ sử dụng các công cụ được cung cấp.""",
    # ------------------------------------------------------------ Figure B.7 --
    planner_part1="""Được đưa ra một truy vấn của người dùng, hãy tạo một chuỗi các lời gọi hàm để giải quyết với khả năng song song hóa tối đa chỉ sử dụng các công cụ sau:
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
– Có nghĩa là: giá trị hiện tại/cuối cùng/mới TRỪ giá trị trước đó/ban đầu/cũ.""",
    # ------------------------------------------------------------ Figure B.9 --
    planner_part2="""Hướng dẫn (tiếp theo):
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
• Đối với các truy vấn tài chính liên quan đến số tiền đô la (ví dụ, $47 triệu), sử dụng giá trị số dưới dạng chuỗi (ví dụ, '47').""",
    # ----------------------------------------------------------- Figure B.11 --
    planner_part3="""Hướng dẫn (cuối cùng):
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
<END_OF_PLAN>""",
    # -------------------------------------- reconstructed, NOT verbatim ------
    planner_query_block="""Truy vấn: {question}
Bối cảnh:
{context}
{subquery_block}Kế hoạch:""",
    subquery_block_header="""BỐI CẢNH BỔ SUNG TỪ TRUY VẤN CON:
{subquery_answers}
""",
)


# ---------------------------------------------------------------- English ---
EN = PromptSet(
    # ---------------------------------------------------- Figure B.2 (G_sq) --
    subquery_generator="""You are an expert at breaking down a user's question into a series of 3 to 5 smaller subqueries for data extraction. Given the user's question and surrounding context (text and tables), your task is to identify the raw data points needed to answer the main question.
Your goal: Generate a list of questions that ONLY EXTRACT raw data values or text or a raw data range.
IMPORTANT INSTRUCTIONS:
• DO NOT generate questions that perform comparisons (e.g., 'which is the lowest/highest').
• DO NOT generate questions that perform calculations (e.g., 'what is the total/average/difference').
• DO NOT generate questions that ask for the final answer.
• Your job is only to ask for the data points themselves. A later step in the pipeline will perform the actual calculations.
Now, perform this task for the following input.
Context:
{context}
Original Question: {question}
Return the subqueries as a JSON object with a single key "subqueries" containing a list of strings.""",
    # ---------------------------------------------------- Figure B.4 (A_sq) --
    subquery_answerer="""You are a helpful assistant. Your task is to answer the given subquery based only on the provided context.
The context consists of the text preceding the table, the table itself, and the text following the table. Always respond with complete sentences, stating the answer to the question or calculation.
Provide a concise and direct answer to the subquery.
Context:
{context}
Subquery:
{subquery}
Answer:""",
    # ------------------------------------------------------------ Figure B.6 --
    planner_system="""You are a numerical reasoning planner, tasked with generating a sequence of function calls to solve a user's query with maximum parallelization, using only the provided tools.""",
    # ------------------------------------------------------------ Figure B.8 --
    planner_part1="""Given a user query, generate a sequence of function calls to solve it with maximum parallelization using only the following tools:
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
– Meaning: the current/final/new value MINUS the previous/initial/old value.""",
    # ----------------------------------------------------------- Figure B.10 --
    planner_part2="""Instructions (continued):
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
• For financial queries involving dollar amounts (e.g., $47 million), use the numeric value as a string (e.g., '47').""",
    # ----------------------------------------------------------- Figure B.12 --
    planner_part3="""Instructions (final):
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
<END_OF_PLAN>""",
    # -------------------------------------- reconstructed, NOT verbatim ------
    planner_query_block="""Query: {question}
Context:
{context}
{subquery_block}Plan:""",
    subquery_block_header="""ADDITIONAL CONTEXT FROM SUBQUERY:
{subquery_answers}
""",
)


def get(lang: str = "vi") -> PromptSet:
    """The Appendix B prompt set for a language."""
    return VI if lang == "vi" else EN


# =============================================================================
# 2. OPT-IN EXTENSIONS -- NOT from the paper, off unless `use_prompt_ext`
# =============================================================================
#
# PERCENT_AS_DECIMAL targets a PA loss the paper itself exhibits.
#
#   Appendix B.9/B.10 tells the planner: given '48.8%', strip the sign and use
#   '48.8'. But ViNumQA gold writes a percentage that is *used as a rate* as a
#   decimal literal -- `divide(333, 0.159)`, `divide(9151, 0.741)`. 34 of the 497
#   gold programs in test.json contain a `0.xx` literal of this kind.
#
#   Following B.9 literally, the planner emits `divide(15.9, 100),
#   divide(333, #0)`. That executes to the right number (EA passes) but fails PA,
#   because scorer.py's symbolic PA admits only literals that occur in the gold
#   program, and neither `15.9` nor `100` does. This is exactly the phenomenon
#   the paper documents in its Example 5.1, and the likely reason its PA (74.07)
#   trails its EA (84.00) by ten points.
#
# EXTRA_OPERATORS completes the operator set.
#
#   The paper's toolset has eight operators plus join(); ViNumQA defines ten.
#   Counted over the real splits, `exp` occurs 0 times in train and 0 in test,
#   `greater` occurs 0 times in train and once in test -- so the paper's eight
#   already cover 496/497 test samples. Enabling this buys at most one sample and
#   widens the space of malformed programs the planner can emit.
#
# PLACEHOLDER_CLARIFICATION targets a real failure measured on a 30-sample
# DeepSeek-V4-Flash smoke run (2026-08-17): 18 of 450 candidates (4%) died at
# transpile with `argument is not numeric: 'giá_trị_mới'` (or 'new' /
# 'giá trị mới') -- the planner echoing back planner_part1's own worked
# example verbatim, `subtract(a='giá_trị_mới', b='giá_trị_cũ')`, instead of
# substituting the real numbers the example's placeholder names stand for.
# The paper's own prompt (verbatim, Appendix B.7/B.8) never says these two
# words are placeholders rather than literal arguments -- this bullet is the
# one-sentence fix, same shape as PERCENT_AS_DECIMAL below.

PERCENT_AS_DECIMAL_VI = """• PHẦN TRĂM DÙNG NHƯ TỶ LỆ: Khi một giá trị phần trăm (ví dụ, '15.9%') được dùng làm tỷ lệ trong phép nhân hoặc phép chia, hãy viết THẲNG giá trị thập phân tương ứng làm hằng số (ví dụ, '0.159'). KHÔNG tạo thêm một bước divide cho '100'."""

PERCENT_AS_DECIMAL_EN = """• PERCENTAGES USED AS RATES: When a percentage value (e.g., '15.9%') is used as a rate in a multiplication or a division, write the corresponding decimal DIRECTLY as a literal (e.g., '0.159'). DO NOT add a separate divide step by '100'."""

EXTRA_OPERATORS_VI = """• Hai công cụ bổ sung, chỉ dùng khi thực sự cần:
– exp(a: str, b: str) -> float: Lũy thừa, a mũ b.
– greater(a: str, b: str) -> str: Trả về 'yes' nếu a > b, ngược lại 'no'."""

EXTRA_OPERATORS_EN = """• Two additional tools, to be used only when genuinely required:
– exp(a: str, b: str) -> float: Exponentiation, a to the power of b.
– greater(a: str, b: str) -> str: Returns 'yes' if a > b, otherwise 'no'."""

PLACEHOLDER_CLARIFICATION_VI = """• LÀM RÕ VÍ DỤ THỨ TỰ PHÉP TRỪ: 'giá_trị_mới' và 'giá_trị_cũ' trong ví dụ subtract(a='giá_trị_mới', b='giá_trị_cũ') CHỈ LÀ TÊN MINH HỌA cho thứ tự tham số, KHÔNG PHẢI giá trị cần viết ra. Bạn PHẢI thay hai tên đó bằng SỐ THẬT lấy từ ngữ cảnh hoặc câu trả lời truy vấn con (ví dụ đúng: subtract(a='96.67', b='100')). TUYỆT ĐỐI KHÔNG được viết nguyên văn 'giá_trị_mới', 'giá_trị_cũ', 'new', hay 'old' làm tham số."""

PLACEHOLDER_CLARIFICATION_EN = """• CLARIFYING THE SUBTRACTION-ORDER EXAMPLE: 'new_value' and 'old_value' in subtract(a='new_value', b='old_value') are ONLY placeholder names showing the argument ORDER, NOT values to write out. You MUST replace them with the ACTUAL NUMBERS from the context or subquery answers (correct example: subtract(a='96.67', b='100')). NEVER write the literal words 'new_value', 'old_value', 'new', or 'old' as an argument."""


def planner_extension(lang: str = "vi") -> str:
    """Extra bullets appended to the planner's Part 2 instructions."""
    if lang == "vi":
        return "\n".join([
            PERCENT_AS_DECIMAL_VI, EXTRA_OPERATORS_VI, PLACEHOLDER_CLARIFICATION_VI,
        ])
    return "\n".join([
        PERCENT_AS_DECIMAL_EN, EXTRA_OPERATORS_EN, PLACEHOLDER_CLARIFICATION_EN,
    ])


# =============================================================================
# 3. THE REPO'S SHARED DIRECT PROMPT -- last-resort fallback only
# =============================================================================
#
# Copied verbatim from
# `notebooks/vinumqa/0-shot/vsf-0-shot-vinumqa-gemma-4-31B-it.ipynb` (cell 4),
# the same SYSTEM_MESSAGE/USER_MESSAGE_FRAME every prompting and SFT notebook in
# this repo uses. Copied rather than imported because a notebook is not
# importable; if the shared prompt changes, it changes in the notebooks and this
# copy follows.
#
# Only reached when the agent pipeline produces no usable candidate at all. It
# exists because an empty prediction scores zero on both metrics with certainty,
# whereas the 0-shot baseline scores well above zero. Every use is recorded in
# the sample's trace as `fallback="direct_prompt"` so the fallback rate can be
# reported alongside PA/EA rather than quietly propping them up.

SYSTEM_MESSAGE = """You are a financial analysis AI. Your task is to generate a sequential computation program to answer the question, based on the provided context.

### LIST OF 10 VALID OPERATORS:

1. add(a, b) -> a + b
2. subtract(a, b) -> a - b
3. multiply(a, b) -> a * b
4. divide(a, b) -> a / b
5. exp(a, b) -> a^b
6. greater(a, b) -> 1.0 if a > b, else 0.0
7. table_sum(row_name, none) -> sum of the numeric values in the table row named `row_name`
8. table_average(row_name, none) -> arithmetic mean of the numeric values in the table row named `row_name`
9. table_max(row_name, none) -> maximum of the numeric values in the table row named `row_name`
10. table_min(row_name, none) -> minimum of the numeric values in the table row named `row_name`

### RULES:
- Do not use free-form mathematical symbols ("+", "-", "*", "/") outside of parentheses. Every calculation must use one of the 10 operators above.
- table_* operators take exactly two arguments: the row name (copied exactly as it appears as the first cell of the target row) and the literal `none` (e.g. table_max(Lãi ròng, none)), never a list of numeric values.
- Do not perform mental calculations or provide explanations. The output must contain only the program string.
- Reference the result of a previous step using #0 (step 1), #1 (step 2), etc. Steps are separated by commas.
- Preserve the original number format from the context. If a value is missing, use 'none'."""

USER_MESSAGE_FRAME = """### CONTEXT:
[TEXT BEFORE TABLE]
{pre_text}

[TABLE]
{table}

[TEXT AFTER TABLE]
{post_text}

### QUESTION:
{question}

### PROGRAM:"""
