# Εκτέλεση Spark στο Google Cloud

Οι οδηγίες βασίζονται στο πείραμα της **27-08-2026**. Για νέα εκτέλεση χρειάζεται έλεγχος των διαθέσιμων εκδόσεων Spark, των δικαιωμάτων πρόσβασης και των τιμών της υπηρεσίας.

Στο πείραμα χρησιμοποιήθηκαν:

- έργο: `har-wisdm-student-2026`·
- περιοχή: `europe-west1`·
- ιδιωτικός χώρος αποθήκευσης (bucket): `gs://har-wisdm-student-2026-data`·
- λογαριασμός υπηρεσίας: `har-spark-runner@har-wisdm-student-2026.iam.gserviceaccount.com`, χωρίς αρχείο κλειδιού·
- περιβάλλον Managed Service for Apache Spark: `2.3` LTS·
- εργασία: `har-watch-10s-20260827-055742`.

## Προετοιμασία λογαριασμού και έργου

Για νέα εγκατάσταση απαιτούνται λογαριασμός Google, χωριστό έργο με ενεργοποιημένη χρέωση και εγκατεστημένο `gcloud`. Πριν από την εκτέλεση ρυθμίζεται ειδοποίηση προϋπολογισμού.

Η σύνδεση στον λογαριασμό και η επιλογή του έργου γίνονται με τις παρακάτω εντολές. Το `STUDENT_PROJECT_ID` αντικαθίσταται από το αναγνωριστικό του νέου έργου:

```bash
gcloud auth login
gcloud auth list
gcloud config set project STUDENT_PROJECT_ID
```

## Ενεργοποίηση των υπηρεσιών

```bash
gcloud services enable \
  dataproc.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  compute.googleapis.com
```

## Χώρος αποθήκευσης και δικαιώματα

Οι ενδεικτικές τιμές αντικαθίστανται με τα στοιχεία της νέας εγκατάστασης:

```bash
PROJECT_ID='STUDENT_PROJECT_ID'
REGION='europe-west1'
BUCKET='STUDENT_UNIQUE_BUCKET'
SERVICE_ACCOUNT='har-spark-runner'
```

Δημιουργείται ιδιωτικό bucket στην επιλεγμένη περιοχή, με ενιαία δικαιώματα πρόσβασης σε επίπεδο bucket και απαγόρευση δημόσιας πρόσβασης. Δημιουργείται επίσης λογαριασμός υπηρεσίας **χωρίς αρχείο κλειδιού**.

Ο λογαριασμός υπηρεσίας χρειάζεται δικαιώματα εκτέλεσης εργασιών Dataproc, εγγραφής στα αρχεία καταγραφής και πρόσβασης στα αντικείμενα του bucket. Ο χρήστης που υποβάλλει την εργασία πρέπει να έχει δικαίωμα χρήσης αυτού του λογαριασμού υπηρεσίας. Οι συγκεκριμένοι ρόλοι της αρχικής εγκατάστασης δίνονται στο `deployment.md`.

Πριν από την ανάθεση ρόλων IAM ελέγχονται το επιλεγμένο έργο και οι λογαριασμοί στους οποίους θα δοθούν τα δικαιώματα.

## Μεταφόρτωση κώδικα και δεδομένων

```text
gs://BUCKET/
├── code/
│   ├── cloud_pipeline.py
│   └── har_spark_cloud_YYYYMMDD.zip
├── raw/watch/
│   ├── accel/*.txt
│   └── gyro/*.txt
└── results/RUN_ID/
```

Το αρχείο ZIP περιέχει το πακέτο `har_spark/` στη ρίζα του. Για το κύριο πείραμα μεταφορτώνονται τα αρχεία επιταχυνσιομέτρου και γυροσκοπίου του ρολογιού.

## Εκκίνηση της εργασίας

Αφού επιβεβαιωθούν η μεταφόρτωση των αρχείων, η περιοχή, ο λογαριασμός υπηρεσίας και η ενεργοποίηση της χρέωσης, η εργασία υποβάλλεται ως εξής:

```bash
gcloud dataproc batches submit pyspark \
  gs://BUCKET/code/cloud_pipeline.py \
  --project=PROJECT_ID \
  --region=europe-west1 \
  --batch=UNIQUE_BATCH_ID \
  --version=2.3 \
  --service-account=SERVICE_ACCOUNT_EMAIL \
  --staging-bucket=BUCKET \
  --py-files=gs://BUCKET/code/har_spark_cloud_YYYYMMDD.zip \
  --labels=purpose=university-har,experiment=watch-fused-10s \
  --properties=spark.dynamicAllocation.initialExecutors=2,spark.dynamicAllocation.minExecutors=2,spark.dynamicAllocation.maxExecutors=4 \
  --async \
  -- \
  --raw-path='gs://BUCKET/raw/watch/*/*.txt' \
  --output-path=gs://BUCKET/results/UNIQUE_RUN_ID \
  --device=watch \
  --window-seconds=10 \
  --minimum-samples=150 \
  --seed=42
```

Η επιλογή `--staging-bucket` δέχεται μόνο το όνομα του bucket. Οι διαδρομές του κώδικα και των δεδομένων χρησιμοποιούν τη μορφή `gs://`.

## Έλεγχος της πορείας εκτέλεσης

```bash
gcloud dataproc batches describe UNIQUE_BATCH_ID \
  --project=PROJECT_ID \
  --region=europe-west1 \
  --format='yaml(state,stateMessage,stateTime,runtimeInfo)'
```

Κάθε εκτέλεση χρησιμοποιεί διαφορετική διαδρομή εξόδου, ώστε να διατηρούνται τα προηγούμενα αποτελέσματα. Αν η εργασία αποτύχει, τα αρχεία καταγραφής και τα ενδιάμεσα αποτελέσματα χρειάζονται για τη διερεύνηση του σφάλματος.

## Έλεγχος μετά την ολοκλήρωση

1. Επιβεβαίωση της τελικής κατάστασης της εργασίας Spark.
2. Έλεγχος των στοιχείων χρέωσης και των ειδοποιήσεων προϋπολογισμού.
3. Λήψη των αποτελεσμάτων που θα διατηρηθούν.
4. Απόφαση για τη διατήρηση ή τη διαγραφή του bucket και του έργου.
5. Ανάκληση προσωρινών δικαιωμάτων IAM όταν δεν χρειάζονται πλέον.

Πριν διαγραφεί οποιοσδήποτε πόρος, ελέγχεται ότι ανήκει στο σωστό έργο και ότι έχει συμφωνηθεί η διαγραφή του με τον κάτοχο του λογαριασμού.
