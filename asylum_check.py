from openai_utils import *
from pydantic import BaseModel
from asylum_ruleset import make_rules
from dataclasses import dataclass

class AnswerReason(BaseModel):
    rule_violation: bool
    missing_info: bool
    said_too_much: bool
    irrelevant_info: bool
    reasoning: str

@dataclass
class FormQuestion:
    question: str
    specific_rules: list[str]
    answer: str
    answer_evaluation: AnswerReason
    finalized: bool

class InformationRequest(BaseModel):
    question: str
    reasoning: str

class InformationRequests(BaseModel):
    requests: list[InformationRequest]

def check_answer(question: FormQuestion) -> FormQuestion:

    system_prompt = (
        "Your job is to determine if the following question is "
        "answered sufficiently. Respond with true or false for "
        "whether the answer is violating one of the rules, "
        "missing info (i.e. not answering the question), saying too "
        "much info, or providing irrelevant information. More than one "
        "can be true. "
        "Here are the rules that the answer must follow:\n"
        f"Rules: {make_rules()}\n"
        f"{"\n ".join(question.specific_rules)}\n"
    )

    system_message = make_message("system", system_prompt)

    user_prompt = (
        f"Question: {question.question}\n"
        f"Answer: {question.answer}\n"
    )
    user_message = make_message("user", user_prompt)

    response: AnswerReason
    response, _, _, reason = call_gpt_formatted(
        [system_message, user_message], AnswerReason
    )

    if response is None:
        print(f"Error from OpenAI:\n {reason}\n")
        question.finalized = False
    
    else:
        question.answer_evaluation = response
        question.finalized = not (response.rule_violation or response.missing_info or response.irrelevant_info or response.said_too_much)

    return question.finalized

def fix_answer(question: FormQuestion) -> FormQuestion:

    system_prompt = (
        "Your job is to determine what additional information is needed " 
        "to make the answer sufficient. Respond with the questions that "
        "need to be answered and the reasoning for why each is needed. "
        "You will also see all of the rules that the answer must follow, "
        "in case those are relevant to the additional information needed. "
        "Here are the rules that the answer must follow:\n"
        f"Rules: {make_rules()}\n"
        f"{"\n ".join(question.specific_rules)}\n"
    )
    system_message = make_message("system", system_prompt)

    user_prompt = (
        f"Question: {question.question}\n"
        f"Answer: {question.answer}\n"
    )
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
            question = check_answer(question)
    return questions

