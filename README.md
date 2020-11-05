# semantic_datasets

В этом репозитории должны находиться попарные сходства. Всё остальное должно быть в других репозиториях.

Данные, описанные в работе Анны Потапенко(http://www.frccsc.ru/sites/default/files/docs/ds/002-073-05/diss/22-potapenko/ds05_22-potapenko_main.pdf) — стр. 107:

I. WordSim353

- Файлы: wordsim353.zip и wordsim353.ipynb
- Язык: английский, есть мультиязычные адаптации (немецкий, итальянский, русский): https://leviants.com/multilingual-simlex999-and-wordsim353/
- Количество слов: 153 строки в формате <слово, слово, оценка>
- Теги: в репозитории без тегов, есть вариант с тегами http://alfonseca.org/eng/research/wordsim353.html (теги следующие: identical tokens, synonym, antonym, hyponym, hyperonym, sibling terms, first is part of the second one, second is part of the first one (at least in one meaning of each), topically related)

- Дополнительно: первоначальным недостатком датасета была нечувствительность к разнице похожих (similarity) и связанных (relatedness) слов. Подробнее в ведении https://arxiv.org/pdf/1408.3456v1.pdf Есть вариант с делением на similar и related: http://alfonseca.org/eng/research/wordsim353.html (см. wordsim_relatedness_goldstandard.txt и wordsim_similarity_goldstandard.txt) 

II. SimLex-999

Файлы: simlex999_rus_without_dupl.csv

III. MEN

Файлы: MEN.zip

IV. Mechanical Turk

Файлы: sim-eval-master.zip

V. HG-RUS

Файлы: russe-evaluation-master.zip

To Do:
Добавить ссылки на происхождение датасетов

Links:
https://rusvectores.org/static/testsets/
