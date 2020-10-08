# Graph representation of RNA

In **2D-Visualization** it is difficult to understand, which section is Represent **stem**, **threeprime**, **fiveprime**, etc. To make it easier, we can generate a **Graph Structure** of RNA. The neato method can take that as input and create a nice visualization of the graph:

---
**ID** : id_001f94081
**Sequence** : GGAAAAGCUCUAAUAACAGGAGACUAGGACUACGUAUUUCUAGGUAACUGGAAUAACCCAUACCAGCAGUUAGAGUUCGCUCUAACAAAAGAAACAACAACAACAAC
**structure** : .....((((((.......)))).)).((.....((..((((((....))))))..)).....))....(((((((....))))))).....................
|2D-Visualization|Graph Structure |
|----|----|
|![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2Ff4f03d6b0b90cba4f89186d430497efb%2F__results___39_0.png?generation=1600167390547492&alt=media) |![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2F0a5f81bd4af44a435e9fbe445c3069c5%2F__results___43_0.png?generation=1600167431782448&alt=media) |

---
**ID** : id_000ae4237
**Sequence**: GGAAACGGGUUCCGCGGAUUGCUGCUAAUAAGAGUAAUCUCUAAAUGCAGCUACCGGCUCUAUAAUGAGCAAAAACGGUAAAUCCCGACAAGCUUGAUUUCGAUCAAGCAAAAGAAACAACAACAACAAC
**structure**: .....((((..((((((...(((((.....((((....))))....)))))..)))((((......)))).....))).....))))....(((((((....))))))).....................
|2D-Visualization|Graph Structure |
|----|----|
|![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2Fe53b559640b7218add3ce4315daf6133%2F__results___53_0.png?generation=1600168784222101&alt=media) | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2F480bf495e1643a28da8298cf8d6fdbe9%2F__results___56_0.png?generation=1600167957368207&alt=media)|

---

**ID** : id_ff691b7e5
**Sequence**: GGAAACUAGCCAUGGGCAGGUUGAAGGUUGGGUGACACUAACUGGAGGACCGAACCGACUAAUGUUGAAAAAUUAGCGGAUGACUUGUGGCGUGUCAGUUCGCUGACACAAAAGAAACAACAACAACAAC
**structure**: ........((((((((..((((...((((((......))))))....))))...(((.(((((........))))))))....))))))))(((((((....))))))).....................
|2D-Visualization|Graph Structure |
|----|----|
|![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2F01423ea4e1a602deb4d7a3c50e5a475f%2F__results___53_0.png?generation=1600169020071954&alt=media)  | ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2907842%2F421d273b06308f5b80f56a167a26b076%2F__results___56_0.png?generation=1600169045301675&alt=media)|

---


**What does this Graph Structure represent?**

* **fiveprime (f)**: The unpaired nucleotides at the 5’ end of a molecule/ chain. Name always starts with ‘f’ (e.g. ‘f0’).

* **threeprime (t)**: The unpaired nucleotides at the 3’ end of a molecule/ chain. Name always start with ‘t’ (e.g. ‘t0’)

* **stem (s)**: Regions of contiguous canonical Watson-Crick base-paired nucleotides. By default, stems have at least 2 consecutive basepairs. Always start with ‘s’ (e.g., ‘s0’, ‘s1’, ‘s2’, …)

* **interior loop (i)**: Bulged out nucleotides and interior loops. An interior loop can contain unpaired bases on either strand or on both strands, flanked by stems on either side. Always start with ‘i’ (‘i0’, ‘i1’, ‘i2’,…)

* **multiloop segment (m)**: Single-stranded regions between two stems. Always start with ‘m’. (‘m0’, ‘m1’, ‘m2’…)

* **hairpin loop (h)**: Always starts with ‘h’.

###  Check My Notebook : [💉OpenVaccine: EDA](https://www.kaggle.com/vatsalparsaniya/openvaccine-eda)

---

* [Reference](https://viennarna.github.io/forgi/graph_tutorial.html)

---
