"""
This file contains variables with prompts to be used when translating the 'yahma/alpaca-cleaned' benchmark:
https://huggingface.co/datasets/yahma/alpaca-cleaned

This file contains only the chosen prompts to be used.
"""

ALPACA_INSTRUCT = """Your task is to translate the given English input, instruction and output into Hebrew. Follow these guidelines:

1. Translate only the input, instruction and output provided.
2. Preserve the original semantic meaning and intent of the texts as accurately as possible in the Hebrew translation.
3. Maintain the exact same formatting as the original English version.
4. Write the translations in a clear and concise style suitable for instruction answering."""
# 5. The translated instructions should be in singular form (Give tips -> תן טיפים).
# 6. The translated output should be with gerund (Eat a meal -> אכילת ארוחה)."""

ALPACA_FEW_SHOTS = """<fewshot examples>
<example>
English:
<output>1. Eat a balanced and nutritious diet: Make sure your meals are inclusive of a variety of fruits and vegetables, lean protein, whole grains, and healthy fats. This helps to provide your body with the essential nutrients to function at its best and can help prevent chronic diseases.

2. Engage in regular physical activity: Exercise is crucial for maintaining strong bones, muscles, and cardiovascular health. Aim for at least 150 minutes of moderate aerobic exercise or 75 minutes of vigorous exercise each week.

3. Get enough sleep: Getting enough quality sleep is crucial for physical and mental well-being. It helps to regulate mood, improve cognitive function, and supports healthy growth and immune function. Aim for 7-9 hours of sleep each night.</output>
<input></input>
<instruction>Give three tips for staying healthy.</instruction>
Hebrew:
<output>1. דיאטה מאוזנת ומזינה: ודא שהארוחות שלך כוללות מגוון של פירות וירקות, חלבון רזה, דגנים מלאים ושומנים בריאים. זה עוזר לספק לגופך את החומרים המזינים החיוניים לתפקוד מיטבי ויכול לסייע במניעת מחלות כרוניות.

2. עיסוק בפעילות גופנית סדירה: פעילות גופנית חיונית לשמירה על עצמות, שרירים ובריאות הלב וכלי הדם. שאף ל-150 דקות לפחות של פעילות אירובית מתונה או 75 דקות של פעילות נמרצת בכל שבוע.

3. הקפדה על שינה מספקת: שינה איכותית ומספקת חיונית לרווחה הפיזית והנפשית. היא עוזרת לווסת את מצב הרוח, לשפר את התפקוד הקוגניטיבי ותומכת בגדילה בריאה ובתפקוד מערכת החיסון. שאף ל-7-9 שעות שינה בכל לילה.</output>
<input></input>
<instruction>תן שלושה טיפים לשמירה על הבריאות.</instruction>
</example>

<example>
English:
<output>The three primary colors are red, blue, and yellow. These colors are called primary because they cannot be created by mixing other colors and all other colors can be made by combining them in various proportions. In the additive color system, used for light, the primary colors are red, green, and blue (RGB).</output>
<input></input>
<instruction>What are the three primary colors?</instruction>
Hebrew:
<output>שלושת צבעי היסוד הם אדום, כחול וצהוב. צבעים אלה נקראים צבעי יסוד מכיוון שלא ניתן ליצור אותם על ידי ערבוב של צבעים אחרים, וניתן ליצור את כל שאר הצבעים על ידי שילובם בפרופורציות שונות. במערכת הצבעים החיבורית, המשמשת לאור, צבעי היסוד הם אדום, ירוק וכחול (RGB).</output>
<input></input>
<instruction>מהם שלושת צבעי היסוד?</instruction>
</example>

<example>
English:
<output>An atom is the basic building block of all matter and is made up of three types of particles: protons, neutrons, and electrons. The structure of an atom can be described as a nucleus at the center surrounded by a cloud of electrons.

The nucleus of an atom is made up of protons and neutrons. Protons are positively charged particles and neutrons are neutral particles with no charge. Both of these particles are located in the nucleus of the atom, which is at the center of the atom and contains most of the atom's mass.

Surrounding the nucleus of the atom is a cloud of electrons. Electrons are negatively charged particles that are in constant motion around the nucleus. The electron cloud is divided into shells or orbitals, and each shell can hold a certain number of electrons. The number of electrons in the outermost shell, called the valence shell, determines the chemical properties of the atom. 

In a neutral atom, the number of protons in the nucleus is equal to the number of electrons in the electron cloud, so the positive and negative charges balance out and the atom has no overall charge. The number of protons, also called the atomic number, determines what element the atom is.</output>
<input></input>
<instruction>Describe the structure of an atom.</instruction>
Hebrew:
<output>אטום הוא אבן הבניין הבסיסית של כל החומרים והוא מורכב משלושה סוגי חלקיקים: פרוטונים, נייטרונים ואלקטרונים. ניתן לתאר את מבנה האטום כגרעין במרכז המוקף בענן של אלקטרונים.

גרעין האטום מורכב מפרוטונים ונייטרונים. פרוטונים הם חלקיקים בעלי מטען חיובי ונייטרונים הם חלקיקים ניטרליים ללא מטען. שני סוגי החלקיקים הללו ממוקמים בגרעין האטום, שנמצא במרכז האטום ומכיל את רוב מסת האטום.

את גרעין האטום מקיף ענן של אלקטרונים. אלקטרונים הם חלקיקים בעלי מטען שלילי הנמצאים בתנועה מתמדת סביב הגרעין. ענן האלקטרונים מחולק לקליפות או אורביטלים, וכל קליפה יכולה להכיל מספר מסוים של אלקטרונים. מספר האלקטרונים בקליפה החיצונית ביותר, הנקראת קליפת הערכיות, קובע את התכונות הכימיות של האטום.

באטום ניטרלי, מספר הפרוטונים בגרעין שווה למספר האלקטרונים בענן האלקטרונים, כך שהמטענים החיוביים והשליליים מתאזנים ולאטום אין מטען כולל. מספר הפרוטונים, הנקרא גם המספר האטומי, קובע איזה יסוד הוא האטום.</output>
<input></input>
<instruction>תאר את מבנה האטום.</instruction>
</example>

<example>
English:
<output>There are several ways to reduce air pollution, including:

1. Reduce energy consumption: By conserving energy, we reduce the amount of pollution emitted from power plants. Some ways to achieve this include: using energy-efficient appliances and lighting, insulating homes, and shutting off lights and electronics when not in use.

2. Use public transportation or carpool: Cars and trucks are major sources of pollution. By using public transportation, carpooling or biking, you can reduce the number of cars on the road, which in turn reduces emissions.

3. Be responsible with waste: When waste is burned or decomposed, they can release harmful gases into the air. Dispose of hazardous waste correctly and recycle materials like plastics, paper, and glass when possible.

4. Support clean energy sources: Renewable energy sources like solar and wind power generate little or no pollution. Support clean energy companies and projects, and consider installing renewable energy systems in your home.

5. Drive efficiently: If you need to drive, make sure your vehicle is well maintained and drive efficiently by avoiding rapid acceleration and braking, and reducing idling.

6. Be mindful of product choices: Some products and practices release more pollution than others. Choose products with eco-friendly packaging and try to minimize your use of aerosol sprays, which can release harmful chemicals into the air.</output>
<input></input>
<instruction>How can we reduce air pollution?</instruction>
Hebrew:
<output>ישנן מספר דרכים להפחית את זיהום האוויר, כולל:

1. הפחתת צריכת אנרגיה: על ידי חיסכון באנרגיה, אנו מפחיתים את כמות הזיהום הנפלטת מתחנות כוח. כמה דרכים להשיג זאת הן: שימוש במכשירים ותאורה חסכוניים באנרגיה, בידוד בתים וכיבוי אורות ומכשירים אלקטרוניים כאשר אינם בשימוש.

2. שימוש בתחבורה ציבורית או בנסיעות משותפות: מכוניות ומשאיות הן מקורות זיהום עיקריים. על ידי שימוש בתחבורה ציבורית, נסיעה משותפת או רכיבה על אופניים, ניתן להפחית את מספר המכוניות על הכביש, מה שבתורו מפחית את הפליטות.

3. התנהלות אחראית עם פסולת: כאשר פסולת נשרפת או מתפרקת, היא עלולה לשחרר גזים מזיקים לאוויר. יש לסלק פסולת מסוכנת כראוי ולמחזר חומרים כמו פלסטיק, נייר וזכוכית ככל האפשר.

4. תמיכה במקורות אנרגיה נקיים: מקורות אנרגיה מתחדשים כמו אנרגיית שמש ורוח מייצרים זיהום מועט או כלל לא. תמכו בחברות ופרויקטים של אנרגיה נקייה, ושקלו להתקין מערכות אנרגיה מתחדשת בביתכם.

5. נהיגה יעילה: אם אתם צריכים לנהוג, ודאו שהרכב שלכם מתוחזק היטב ונהגו ביעילות על ידי הימנעות מהאצה ובלימה מהירות, והפחתת עמידה במקום עם מנוע פועל.

6. מודעות לבחירת מוצרים: מוצרים מסוימים ושיטות מסוימות פולטים יותר זיהום מאחרים. בחרו מוצרים עם אריזות ידידותיות לסביבה ונסו למזער את השימוש בתרסיסים, העלולים לשחרר כימיקלים מזיקים לאוויר.</output>
<input></input>
<instruction>כיצד אנו יכולים להפחית את זיהום האוויר?</instruction>
</example>

<example>
English:
<output>I had to make a difficult decision when I was working as a project manager at a construction company. I was in charge of a project that needed to be completed by a certain date in order to meet the client’s expectations. However, due to unexpected delays, we were not able to meet the deadline and so I had to make a difficult decision. I decided to extend the deadline, but I had to stretch the team’s resources even further and increase the budget. Although it was a risky decision, I ultimately decided to go ahead with it to ensure that the project was completed on time and that the client’s expectations were met. The project was eventually successfully completed and this was seen as a testament to my leadership and decision-making abilities.</output>
<input></input>
<instruction>Pretend you are a project manager of a construction company. Describe a time when you had to make a difficult decision.</instruction>
Hebrew:
<output>נאלצתי לקבל החלטה קשה כשעבדתי כמנהל פרויקטים בחברת בנייה. הייתי אחראי על פרויקט שהיה צריך להסתיים עד תאריך מסוים כדי לעמוד בציפיות הלקוח. עם זאת, עקב עיכובים בלתי צפויים, לא הצלחנו לעמוד בלוח הזמנים ולכן נאלצתי לקבל החלטה קשה. החלטתי להאריך את המועד האחרון, אבל נאלצתי למתוח את משאבי הצוות עוד יותר ולהגדיל את התקציב. למרות שזו הייתה החלטה מסוכנת, בסופו של דבר החלטתי להמשיך איתה כדי להבטיח שהפרויקט יושלם בזמן ושהציפיות של הלקוח יתממשו. הפרויקט הושלם בסופו של דבר בהצלחה וזה נתפס כעדות ליכולות המנהיגות וקבלת ההחלטות שלי.</output>
<input></input>
<instruction>העמד פנים שאתה מנהל פרויקטים בחברת בנייה. תאר מקרה בו נאלצת לקבל החלטה קשה.</instruction>
</example>
</fewshot examples>"""