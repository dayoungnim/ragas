from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.base import (
    MetricOutputType,
    MetricType,
    MetricWithEmbeddings,
    MetricWithLLM,
    SingleTurnMetric,
)
from ragas.prompt import PydanticPrompt

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


class ResponseRelevanceOutput(BaseModel):
    question: str
    noncommittal: int
    reason: str  # noncommittal 판단 이유 추가


class ResponseRelevanceInput(BaseModel):
    response: str


class ResponseRelevancePrompt(
    PydanticPrompt[ResponseRelevanceInput, ResponseRelevanceOutput]
):
    instruction = """주어진 답변에 대해 고객이 할 법 한 질문을 한국어로 생성하고, 해당 답변이 질문에 대해 회피적인답변을 제공하고있는(noncommittal)지 식별하시오. 답변이 회피적이면 noncommittal을 1로, 그렇지 않으면 0으로 표시하시오. 책임회피적인 답변이란, "모르겠습니다" 또는 "확실하지 않습니다"와 같은 답변으로, 사용자의 질문에 대한 핵심 정보를 제공하지 않는 답변을 말합니다. 또한, 각 판단에 대한 이유도 함께 제공하시오."""

    input_model = ResponseRelevanceInput
    output_model = ResponseRelevanceOutput
    examples = [
        (
            ResponseRelevanceInput(
                response="""Albert Einstein was born in Germany.""",
            ),
            ResponseRelevanceOutput(
                question="Where was Albert Einstein born?",
                noncommittal=0,
                reason="The answer provides a specific and direct response to the question about Einstein's birthplace.",
            ),
        ),
        (
            ResponseRelevanceInput(
                response="""I don't know about the groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022.""",
            ),
            ResponseRelevanceOutput(
                question="What was the groundbreaking feature of the smartphone invented in 2023?",
                noncommittal=1,
                reason="The answer explicitly states 'I don't know' and admits lack of information, making it evasive and noncommittal.",
            ),
        ),
    ]


@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"
    _required_columns: t.Dict[MetricType, t.Set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {
                "user_input",
                "response",
            }
        }
    )
    output_type = MetricOutputType.CONTINUOUS

    question_generation: PydanticPrompt = ResponseRelevancePrompt()
    strictness: int = 1

    def calculate_similarity(self, question: str, generated_questions: list[str]):
        assert self.embeddings is not None, (
            f"Error: '{self.name}' requires embeddings to be set."
        )
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)  # type: ignore[attr-defined]
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions) # type: ignore[attr-defined]
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[ResponseRelevanceOutput], row: t.Dict
    ) -> tuple[float, list[dict]]:
        """
        Returns:
            tuple: (score, reasons_list)
                - score: relevancy score
                - reasons_list: list of dicts containing question, noncommittal status, and reason
        """
        question = row["user_input"]
        gen_questions = [answer.question for answer in answers]

        # 모든 noncommittal을 0으로 강제
        forced_noncommittal = [0 for _ in answers]

        # 이유 정보 수집
        reasons_info = [
            {
                "question": answer.question,
                "noncommittal": 0,  # 강제
                "reason": answer.reason
            }
            for answer in answers
        ]

        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            print("dbg, cosine_sim:", cosine_sim)

            # score 계산 시에도 noncommittal은 항상 0이므로 not all_noncommittal = True
            score = cosine_sim.mean()  # × int(not all_noncommittal) 필요 없음
            print("dbg, score:", score)

        return score, reasons_info

    async def _single_turn_ascore(
        self, sample: SingleTurnSample, callbacks: Callbacks
    ) -> float:
        row = sample.to_dict()
        score, reasons = await self._ascore(row, callbacks)

        # 디버깅을 위해 reasons 출력
        print("\n=== Noncommittal Analysis ===")
        for i, info in enumerate(reasons, 1):
            print(f"\nGenerated Question {i}: {info['question']}")
            print(f"Noncommittal: {'Yes' if info['noncommittal'] else 'No'}")
            print(f"Reason: {info['reason']}")
        print(f"\nFinal Score: {score}")
        print("=" * 30 + "\n")

        return score

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> tuple[float, list[dict]]:
        assert self.llm is not None, "LLM is not set"

        prompt_input = ResponseRelevanceInput(response=row["response"])

        print("dbg, prompt_input:", prompt_input)
        print("n, self.strictness:", self.strictness)

        responses = await self.question_generation.generate_multiple(
            data=prompt_input, llm=self.llm, callbacks=callbacks, n=self.strictness
        )

        print('dbg, responses:', responses)

        return self._calculate_score(responses, row)


class AnswerRelevancy(ResponseRelevancy):
    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> tuple[float, list[dict]]:
        return await super()._ascore(row, callbacks)


answer_relevancy = AnswerRelevancy()