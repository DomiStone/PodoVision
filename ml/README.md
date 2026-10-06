# PodoVision ML

## Ziel
Zweistufiges System:
1. Läsionssegmentierung/Lokalisierung.
2. Klassifikation der lokalisierten klinischen Smartphone-Aufnahme.

PAD-UFES-20 wird für die Smartphone-Klassifikation verwendet. Der Split erfolgt strikt nach patient_id, damit Bilder derselben Person nicht in Training und Test landen.

## Klassifikator trainieren
PAD-UFES-20 unter data/pad entpacken, dann:

```
pip install -r ml/requirements.txt
PAD_ROOT=data/pad python ml/train_classifier.py
```

Erzeugt:
- web/model/classifier.onnx
- web/model/classes.json
- web/model/metrics.json

## Safety Gate
Das Modell wird NICHT automatisch als medizinisch brauchbar bezeichnet. Vor Integration müssen mindestens klassenweise Sensitivität/Spezifität, Konfusionsmatrix, Kalibrierung und externe Validierung geprüft werden. Ein hoher Gesamtscore reicht nicht.

## Segmentierung
ISIC Task 1 kann als Startpunkt für eine Segmentierungs-KI dienen. Da diese Bilder dermatoskopisch sind, muss die Segmentierung anschließend auf klinischen Smartphone-Aufnahmen extern geprüft bzw. mit klinischen Masken nachtrainiert werden.
