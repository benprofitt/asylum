from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import uuid
import os
import time
import json
from datetime import timedelta

from asylum_check import check_answers_and_give_feedback, FormQuestion

IS_PRODUCTION = os.getenv("ENV") == "production"
FRONTEND_URL = os.getenv("FRONTEND_URL")
API_URL = os.getenv("API_URL")

app = Flask(__name__)
print("FRONTEND_URL", FRONTEND_URL)
# CORS(app, supports_credentials=True, origins=FRONTEND_URL)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins


@app.route("/")
def main():
    return jsonify({"message": "Hello, World!"})


@app.route("/grade", methods=["POST"])
def grade():
    data = request.get_json()

    print(data)

    if "questions" in data:

        # Create FormQuestion objects from the json
        form_questions = [
            FormQuestion(q["question"], q["specific_rules"], q["answer"], None, False)
            for q in data["questions"]
        ]

        # Verify the answers
        feedback = check_answers_and_give_feedback(form_questions)

        # Return the listing in JSON
        return jsonify(feedback)
    else:
        return jsonify({"error": "Invalid input. Expected a list of file paths."}), 400