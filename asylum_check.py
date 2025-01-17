from openai_utils import *
from pydantic import BaseModel
from asylum_ruleset import make_rules_short_answer, make_cover_letter_rules
from dataclasses import dataclass

import json, sys


class AnswerReason(BaseModel):
    rule_violation: bool
    missing_info: bool
    said_too_much: bool
    irrelevant_info: bool
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "rule_violation": self.rule_violation,
            "missing_info": self.missing_info,
            "said_too_much": self.said_too_much,
            "irrelevant_info": self.irrelevant_info,
            "reasoning": self.reasoning,
        }


@dataclass
class CoverLetter:
    body: str
    specific_rules: list[str]
    answer_evaluation: AnswerReason
    finalized: bool


@dataclass
class FormQuestion:
    question: str
    specific_rules: list[str]
    answer: str
    answer_evaluation: AnswerReason
    finalized: bool


def make_question_from_json(json_question: dict) -> FormQuestion:
    return FormQuestion(
        question=json_question["question"],
        specific_rules=json_question["specific_rules"],
        answer=json_question["answer"],
        answer_evaluation=None,
        finalized=False,
    )


class InformationRequest(BaseModel):
    question: str
    reasoning: str


class InformationRequests(BaseModel):
    requests: list[InformationRequest]


def check_answer(question: FormQuestion) -> bool:

    if question.specific_rules is not None and len(question.specific_rules) > 0:
        rules = "\n ".join(question.specific_rules)
    else:
        rules = make_rules_short_answer()

    system_prompt = (
        "Your job is to determine if the following question is "
        "answered sufficiently. Respond with true or false for "
        "whether the answer is violating one of the rules, "
        "missing info (i.e. not answering the question), saying too "
        "much info, or providing irrelevant information. More than one "
        "can be true. "
        "Here are the rules that the answer should follow:\n"
        f"Rules: {rules}\n"
        "End of rules."
        "Not every rule must be followed if that would cause contridiction. "
        "For example, if the answer is short and references back to the cover letter, "
        "it is not necessary to satisfy all other inclusion-specific rules "
        "because they are likely satisfied by the cover letter. "
    )

    system_message = make_message("system", system_prompt)

    user_prompt = f"Question: {question.question}\n" f"Answer: {question.answer}\n"
    user_message = make_message("user", user_prompt)

    response: AnswerReason
    response, _, _, reason = call_gpt_formatted([system_message, user_message], AnswerReason)

    if response is None:
        print(f"Error from OpenAI:\n {reason}\n")
        question.finalized = False

    else:
        question.answer_evaluation = response
        question.finalized = not (
            response.rule_violation
            or response.missing_info
            or response.irrelevant_info
            or response.said_too_much
        )

    return question.finalized


def create_info_requests(question: FormQuestion) -> list[InformationRequest]:

    specific_rules = "\n ".join(question.specific_rules)
    system_prompt = (
        "Your job is to determine what additional information is needed "
        "to make the answer sufficient. Respond with the questions that "
        "need to be answered and the reasoning for why each is needed. "
        "You will also see all of the rules that the answer must follow, "
        "in case those are relevant to the additional information needed. "
        "Be explicit about what information is missing or needed. "
        "Here are the rules that the answer must follow:\n"
        f"Rules: {make_rules_short_answer()}\n"
        f"{specific_rules}\n"
    )
    system_message = make_message("system", system_prompt)

    user_prompt = f"Question: {question.question}\n" f"Answer: {question.answer}\n"
    user_message = make_message("user", user_prompt)

    messages = [system_message, user_message]

    response: InformationRequests
    response, _, _, reason = call_gpt_formatted(messages, InformationRequests)

    if response is None:
        print(f"Error from OpenAI:\n {reason}\n")
        requests = []

    else:
        requests = response.requests

    return requests


def check_all_answers(questions: list[FormQuestion]) -> list[FormQuestion]:
    for question in questions:
        if not question.finalized:
            check_answer(question)
    return questions


def reduce_requests(requests: list[InformationRequest]) -> InformationRequests:

    system_prompt = (
        "Your job is to determine if the following information requests "
        "can be combined to reduce redundancy. Respond with the reduced "
        "list of information requests. "
    )
    system_message = make_message("system", system_prompt)

    req_string = "\n ".join([r.question for r in requests])
    user_prompt = (
        "The following information requests have been made. "
        "Please determine if any can be combined to reduce redundancy. "
        "If so, provide the reduced list of information requests. "
        f"Requests: {req_string}\n"
    )
    user_message = make_message("user", user_prompt)

    messages = [system_message, user_message]

    response: InformationRequests
    response, _, _, reason = call_gpt_formatted(messages, InformationRequests)

    if response is None:
        print(f"Error from OpenAI:\n {reason}\n")
        reduced_requests = []

    else:
        reduced_requests = response.requests

    return reduced_requests


def serve_requests_to_user_input(requests: list[InformationRequest]) -> list[str]:
    answers = {}
    for request in requests:
        answer = input(f"{request.question}\n")
        answers[request.question] = answer

    return answers


def verify_answers(questions: list[FormQuestion]) -> list[FormQuestion]:

    # Check all answers
    questions = check_all_answers(questions)

    # Create information requests for all questions that are not finalized
    # and are missing information
    requests = []
    for question in questions:
        if not question.finalized and question.answer_evaluation.missing_info:
            requests.extend(create_info_requests(question))

    # Combine the requests with each other to reduce redundancy
    reduced_requests = reduce_requests(requests)

    new_information: dict = serve_requests_to_user_input(reduced_requests)

    # Print new info to check if it is correct
    for key, value in new_information.items():
        print(f"{key}: {value}")


def check_answers_and_give_feedback(questions: list[FormQuestion]) -> list:
    # Check all answers
    questions = check_all_answers(questions)

    # Create a dict to store the feedback and show it to the original user
    feedback = []
    for question in questions:
        d = {"question": question.question, "answer": question.answer}
        d["evaluation"] = question.answer_evaluation.to_dict()
        feedback.append(d)

    return feedback


def check_full_cover_letter(letter: CoverLetter) -> dict:
    # check cover letter
    q: FormQuestion = FormQuestion(
        question="Write a cover letter for an asylum application.",
        specific_rules=make_cover_letter_rules(),
        answer=letter.body,
        answer_evaluation=None,
        finalized=False,
    )

    # check the answer
    check_answer(q)

    feedback = {}
    feedback["question"] = q.question
    feedback["answer"] = q.answer
    feedback["evaluation"] = q.answer_evaluation.to_dict()

    return feedback


def main():
    # read in questions from a file
    questions = {}
    with open(sys.argv[1], "r") as f:
        questions = json.load(f)

    # create FormQuestion objects from the json
    form_questions = [make_question_from_json(q) for q in questions]

    # verify the answers
    verify_answers(form_questions)
