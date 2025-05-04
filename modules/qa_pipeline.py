from transformers import pipeline

class QAPipeline:
    def __init__(self):
        self.qa_model = pipeline("question-answering", model="deepset/roberta-base-squad2", device=-1)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

    def get_answer(self, question, context):
        result = self.qa_model(question=question, context=context)
        raw_answer = result['answer']

        # Combine question, answer, and context
        input_text = f"Question: {question}\nAnswer: {raw_answer}\n\nContext: {context}"

        # Truncate input_text to avoid tokenizer overflow (limit ~1024 tokens ≈ 1000 characters)
        input_text = input_text[:1000]

        try:
            summary = self.summarizer(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Summarization failed: {e}. Showing raw answer instead:\n\n{raw_answer}"
        return summary
