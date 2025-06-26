# Instruction prompts
ARC_INSTRUCT = """Your task is to translate the given English question and possible answers into Hebrew. Follow these guidelines:

1. Only translate the question and answer options provided. Do not add any additional text or instructions.
2. Preserve the original semantic meaning and intent of the question and answers as accurately as possible in the Hebrew translation.
3. Maintain the same formatting as the original English version, including any bullet points, numbering, or other formatting elements.
4. Provide the Hebrew translation immediately after these instructions, without any preamble or additional context."""

# Format prompts (to be used in the prompt when needed)
ARC_FORMAT = """<response_format>
<question>Translated question</question>
<option 1>Translated answer option 1</option 1>
<option 2>Translated answer option 2</option 2>
<option 3>Translated answer option 3</option 3>
<option 4>Translated answer option 4</option 4>
</response_format>"""

# The few-shots samples
ARC_FEW_SHOTS = """<fewshot_examples>

<example>
English:
<question>George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?</question>
<option 1>dry palms</option 1>
<option 2>wet palms</option 2>
<option 3>palms covered with oil</option 3>
<option 4>palms covered with lotion</option 4>
Hebrew:
<question>ג'ורג' רוצה לחמם את ידיו במהירות על ידי שפשופן. איזה משטח עור ייצר את החום הרב ביותר?</question>
<option 1>כפות ידיים יבשות</option 1>
<option 2>כפות ידיים רטובות</option 2>
<option 3>כפות ידיים מכוסות בשמן</option 3>
<option 4>כפות ידיים מכוסות בקרם</option 4>
</example>

<example>
English:
<question>Which of the following statements best explains why magnets usually stick to a refrigerator door?</question>
<option 1>The refrigerator door is smooth.</option 1>
<option 2>The refrigerator door contains iron.</option 2>
<option 3>The refrigerator door is a good conductor.</option 3>
<option 4>The refrigerator door has electric wires in it.</option 4>
Hebrew:
<question>איזה מההיגדים הבאים מסביר בצורה הטובה ביותר מדוע מגנטים בדרך כלל נדבקים לדלת המקרר?</question>
<option 1>דלת המקרר חלקה.</option 1>
<option 2>דלת המקרר מכילה ברזל.</option 2>
<option 3>דלת המקרר מוליכה טוב.</option 3>
<option 4>דלת המקרר מכילה חוטי חשמל.</option 4>
</example>

<example>
English:
<question>A fold observed in layers of sedimentary rock most likely resulted from the</question>
<option 1>cooling of flowing magma.</option 1>
<option 2>converging of crustal plates.</option 2>
<option 3>deposition of river sediments.</option 3>
<option 4>solution of carbonate minerals.</option 4>
Hebrew:
<question>קפל שנצפה בשכבות של סלע משקע נוצר ככל הנראה כתוצאה מ</question>
<option 1>התקררות של מגמה זורמת.</option 1>
<option 2>התכנסות של לוחות קרום כדור הארץ.</option 2>
<option 3>שקיעה של משקעי נהר.</option 3>
<option 4>המסה של מינרלים פחמתיים.</option 4>
</example>

<example>
English:
<question>Which of these do scientists offer as the most recent explanation as to why many plants and animals died out at the end of the Mesozoic era?</question>
<option 1>worldwide disease</option 1>
<option 2>global mountain building</option 2>
<option 3>rise of mammals that preyed upon plants and animals</option 3>
<option 4>impact of an asteroid created dust that blocked the sunlight</option 4>
Hebrew:
<question>איזה מההסברים הבאים מציעים מדענים כהסבר העדכני ביותר לכך שצמחים ובעלי חיים רבים נכחדו בסוף עידן המזוזואיקון?</question>
<option 1>מחלה עולמית</option 1>
<option 2>בניית הרים גלובלית</option 2>
<option 3>עליית יונקים שטרפו צמחים ובעלי חיים</option 3>
<option 4>פגיעת אסטרואיד יצרה אבק שחסם את אור השמש</option 4>
</example>

<example>
English:
<question>A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat travels northeast. In which direction is the wind most likely applying force to the sails of the boat?</question>
<option 1>west</option 1>
<option 2>east</option 2>
<option 3>north</option 3>
<option 4>south</option 4>
Hebrew:
<question>סירה מושפעת מזרם נהר הזורם צפונה ומרוח הנושבת על מפרשיה. הסירה נעה לכיוון צפון-מזרח. באיזה כיוון הרוח ככל הנראה מפעילה כוח על מפרשי הסירה?</question>
<option 1>מערב</option 1>
<option 2>מזרח</option 2>
<option 3>צפון</option 3>
<option 4>דרום</option 4>
</example>


</fewshot_examples>"""