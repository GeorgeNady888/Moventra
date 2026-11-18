import json
import tempfile
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

import storage
from service import run_analysis

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # refuse uploads > 50 MB

CONDITIONS = ("rest", "postural", "kinetic")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test_page():
    """Temporary page to try the API from the browser (step 2 only)."""
    return render_template("test.html")


@app.route("/api/health")
def health():
    return jsonify(status="ok")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Receive a recording CSV, return the analysis as JSON."""
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify(error="No file uploaded. Send a CSV in the 'file' field."), 400

    condition = request.form.get("condition", "rest").lower()
    if condition not in CONDITIONS:
        return jsonify(error=f"condition must be one of {CONDITIONS}"), 400

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.csv"
        file.save(path)
        try:
            result = run_analysis(path, condition)
        except Exception as e:           # bad/short/incomplete recording -> readable message
            return jsonify(error=str(e)), 422
        try:
            meta = json.loads(request.form.get("meta", "{}"))
            if not isinstance(meta, dict):
                meta = {}
        except ValueError:
            meta = {}
        try:                              # a storage problem must never lose the result
            result["record_id"], result["created_at"] = storage.save(result, meta, path.read_text(encoding="utf-8"))
        except Exception as e:
            result["record_id"], result["save_error"] = None, str(e)
        return jsonify(result)


@app.route("/records/<int:rid>/report")
def record_report(rid):
    """Printable report page: the browser's Print > Save as PDF makes the PDF."""
    rec = storage.get(rid)
    if rec is None:
        return "Record not found", 404
    return render_template("report.html", rec=rec)


@app.route("/api/records")
def api_records():
    return jsonify(storage.list_records())


@app.route("/api/records/<int:rid>")
def api_record(rid):
    rec = storage.get(rid)
    return (jsonify(rec), 200) if rec else (jsonify(error="Record not found"), 404)


@app.route("/api/records/<int:rid>/csv")
def api_record_csv(rid):
    got = storage.get_csv(rid)
    if got is None:
        return jsonify(error="Record not found"), 404
    text, condition, created = got
    name = f"tremor_{condition}_{created[:19].replace(':', '-')}.csv"
    return Response(text, mimetype="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.route("/api/records/<int:rid>", methods=["DELETE"])
def api_record_delete(rid):
    return (jsonify(ok=True), 200) if storage.delete(rid) else (jsonify(error="Record not found"), 404)


if __name__ == "__main__":
    app.run(debug=True)