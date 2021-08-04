# fed-ml-prototype
## Beschreibung des Projekts
Ziel dieses Projekts ist es, ein Federated Learning (FL) Szenario für einen Use Case aus der Industrie zu implementieren sowie die Performance dieser Implementierung mit möglichen Alternativen zu vergleichen. 

Federated Learning ist eine Methode des maschinellen Lernens, bei welcher ein Modell über dezentrale Entitäten (Clients) hinweg trainiert wird. Hierbei werden, basierend auf den verfügbaren Daten der einzelnen Entitäten, lokale Modelle trainiert, ohne die zugrundeliegenden Daten auszutauschen. Im Anschluss werden die Parameter der einzelnen Modelle an eine zentrale Einheit (Server) gesendet, welche die erhaltenen Parameter nutzt, um ein globales Modell zu generieren. 

Federated Learning steht somit im Gegensatz zu herkömmlichen, zentralisierten Methoden des maschinellen Lernens, bei denen alle lokalen Datensätze auf einen Server hochgeladen werden, um ein Modell zu trainieren.
## Datensatz
Für die Implementierung wurde der Datensatz *Industrial Quality Control of Packages* ausgewählt, welcher auf [Kaggle](https://www.kaggle.com/christianvorhemus/industrial-quality-control-of-packages) verfügbar ist. Dieser Datensatz besteht aus 400 synthetisch erstellten, gelabelten Bildern von Produktverpackungen, wovon 200 beschädigt sind während die restlichen 200 Verpackungen unversehrt sind. Beide Kategorien bestehen zu einer Hälfte aus Bildern, welche die Verpackungen von der Seite zeigen, wohingegen die andere Hälfte die Verpackungen von oben zeigt. Das Ziel ist es, ein Modell zu trainieren, welches automatisch erkennt, ob eine Verpackung beschädigt ist oder nicht.

Die nachfolgenden Bilder zeigen die Draufsicht sowie die Seitenansicht einer der beschädigten Verpackungen.

<p align="middle">
  <img src="./data_sample/0915598067788_top.png" width="360" /> 
  <img src="./data_sample/0915598067788_side.png" width="360" />
</p>
## Ansatz
Um den Datensatz für Federated Learning nutzen zu können, wurde dieser in drei lokale Subdatensätze à 120 Bilder aufgeteilt, welche jeweils zu gleichen Teilen aus intakten und beschädigten Verpackungen bestehen. Die verbliebenen 40 Bilder wurden als globaler Testdatensatz verwendet, um die Performance der einzelnen Modelle objektiv miteinander vergleichen zu können.

Basierend auf den lokalen Datensätzen wurde ein Federated Learning Szenario mit drei Clients und acht Runden implementiert. Um aus den lokal trainierten Modellen ein globales Modell zu generieren, wurde der Federated Averaging (FedAvg) Algorithmus von McMahan et al. (2016) genutzt. Die genauen Parameter finden sich in `server.py` und können für weitere Experimente angepasst werden.

Um die Performanz des Federated Learning Ansatzes einschätzen zu können, wurden folgende Modelle als Baselines verwendet:

* Ein globales, zentralisiertes Modell für welches alle Daten der drei Clients als Trainingsdaten verwendet wurden
* Ein lokales Modell je Client, basierend auf den jeweiligen lokalen Trainingsdaten

Das globale Modell stellt dabei den klassichen Fall dar, in welchem alle verfügbaren Daten zentral gesammelt werden, was jedoch aufgrund datenschutztechnischer Umstände oftmals nicht umsetzbar ist. Im Gegensatz dazu stehen die lokalen Modelle, für welche jede Entität nur die ihr lokal verfügbaren Daten verwendet. Hierbei werden keine Daten zwischen den einzelnen Entitäten ausgetauscht, allerdings steht den einzelnen Modellen jeweils nur ein Teil des Gesamtdaten zur Verfügung, was somit zu Einbußen in der Güte der Modelle im Vergleich zu einem zentralisierten Modell führen kann. Federated Learning versucht, den Verlust an Performanz gegenüber einem zentralisierten Modell gering zu halten, indem basierend auf den Parametern der lokalen Modelle ein globales Modell generiert wird, ohne dabei jedoch die zugrundeliegenden Daten auszutauschen und offenzulegen.

Um Vergleichbarkeit zu sichern, wurde für alle Varianten dasselbe neuronale Netz verwendet. Hierbei wurde ein auf dem ImageNet Datensatz vortrainiertes EfficientNet-B0 gewählt, welches im Anschluss auf den jeweils verfügbaren Daten nachtrainiert wurde (Fine-tuning Ansatz).
## Code
Der Code für das globale, zentralisierte Modell sowie die drei lokalen Modelle ist im Notebook `Baselines.ipynb` zu finden und kann beispielsweise über Google Colab ausgeführt werden. 

Die Implementierung des Federated Learning Ansatzes nutzt [Flower](https://github.com/adap/flower) (`flwr`), ein einfach zu benutzendes Framework für Federated Learning Systeme und basiert auf dem auf GitHub verfügbaren [TensorFlow Tutorial](https://github.com/adap/flower/tree/main/examples/advanced_tensorflow). Um die Implementierung zu starten, muss `server.py`, gefolgt von `client1.py`, `client2.py` und `client3.py` ausgeführt werden. Alternativ kann der Federated Learning Ansatz auch über `run.sh` gestartet werden.

Bevor die Python Programme ausgeführt werden können, müssen jedoch erst die erforderlichen Pakete installiert werden, welche in `requirements.txt` spezifiziert sind.

## Ergebnisse
In jedem Szenario wurde ein 80/20 Split der Daten in Trainings- und Validierungsdaten vorgenommen. Die folgende Tabelle zeigt die Accuracy der Baseline Modelle auf den jeweiligen Validierungsdatensätzen sowie dem globalen Testdatensatz. 

<table>
<thead>
  <tr>

    <th colspan="5" style="text-align: center">Baselines</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center"></td>
    <td style="text-align: center">Globales, zentralisiertes Modell</td>
    <td style="text-align: center">Lokales Modell Client 1</td>
    <td style="text-align: center">Lokales Modell Client 2</td>
    <td style="text-align: center">Lokales Modell Client 3</td>
  </tr>
  <tr>
     <td style="text-align: center">Accuracy auf jeweiligem Validierungsdatensatz</td>
    <td style="text-align: center">75,00%</td>
    <td style="text-align: center">45,83%</td>
    <td style="text-align: center">58,33%</td>
    <td style="text-align: center">75,00%</td>
  </tr>
  <tr>
    <td style="text-align: center">Accuracy auf globalem Testdatensatz</td>
    <td style="text-align: center">75,00%</td>
    <td style="text-align: center">62,50%</td>
    <td style="text-align: center">62,50%</td>
    <td style="text-align: center">57,50%</td>
  </tr>
</tbody>
</table>

Aufgrund von Randomisierungen während des Trainings variieren die genauen Werte bei jeder Ausführung, weshalb für eine verlässlichere, statistische Auswertung der Performanz der Modelle mehrere Trainingsdurchläufe absolviert werden sollten. Dennoch lässt sich eine Tendenz erkennen, dass das globale, zentralisierte Modell ein besseres Ergebnis liefert als die lokalen Modelle der einzelnen Clients, welche beim Trainieren nur auf ihren jeweiligen Teil der verfügbaren Daten zugreifen können. Mit einer maximalen Accuracy von 75,00% zeigt sich jedoch, dass die Modelle Schwierigkeiten bei der zuverlässigen Erkennung von Beschädigungen der Verpackungen haben, was vermutlich auf die geringe Größe des Datensatzes zurückzuführen ist.

Die nachfolgende Tabelle zeigt die Accuracy des Federated Learning Modells nach acht Runden auf den Validierungsdatensätzen der einzelnen Clients sowie dem globalen Testdatensatz.

<table>
<thead>
  <tr>

    <th colspan="2" style="text-align: center">Federated Learning</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td style="text-align: center"></td>
    <td style="text-align: center">Federated Learning Modell</td>

  </tr>
  <tr>
    <td style="text-align: center">Accuracy auf lokalem Validierungsdatensatz von Client 1</td>
    <td style="text-align: center">70,83%</td>
  </tr>
  <tr>
    <td style="text-align: center">Accuracy auf lokalem Validierungsdatensatz von Client 2</td>
    <td style="text-align: center">62,50%</td>
  </tr>
  <tr>
    <td style="text-align: center">Accuracy auf lokalem Validierungsdatensatz von Client 3</td>
    <td style="text-align: center">87,50%</td>
  </tr>
  <tr>
    <td style="text-align: center">Accuracy auf globalem Testdatensatz</td>
    <td style="text-align: center">65,00%</td>
  </tr>
</tbody>
</table>

Die Ergebnisse zeigen, dass das Federated Learning Modell nicht ganz an die Performanz des globalen, zentralisierten Modells herankommt, jedoch im Vergleich zu den lokal trainierten Modellen sowohl eine höhere Accuracy auf den lokalen Validierungsdatensätzen der einzelnen Clients als auch auf dem globalen Testdatensatz erzielen kann. 
## Diskussion
Eine Mehrfachausführung des Codes sowie eine statistische Analyse der Ergebnisse ist für den Erhalt verlässlicher Ergebnisse zu empfehlen. Darüber hinaus stellen Durchführungen mit ungleichmäßig verteilten Daten ein interessantes Experiment dar, da aktuell jeder Subdatensatz zu gleichen Teilen aus Bildern von intakten sowie beschädigten Verpackungen besteht. Des Weiteren ist der genutzte Datensatz mit einer Gesamtanzahl von 400 Bildern vergleichsweise klein, weshalb die Ergebnisse je nach Zusammensetzung der Trainings-, Validierungs- und Testdaten schwanken.

Nichtsdestotrotz zeigen die vorliegenden Ergebnisse, dass Federated Learning verglichen mit einer ausschließlichen Nutzung von lokal verfügbaren Datensätzen das Potenzial besitzt, eine Verbesserung der Qualität von Machine Learning Modellen zu erzielen. Somit stellt Federated Learning einen wichtigen Ansatz in Fällen dar, in denen eine Zentralisierung der verteilten Daten nicht möglich ist.