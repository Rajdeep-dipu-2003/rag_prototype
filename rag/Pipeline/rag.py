from rag.Pipeline.Models.flash import create_model
from langchain_core.prompts import ChatPromptTemplate
from rag.Pipeline.retriver import get_documents


class RAGPipeline:
    def __init__(self):
        self.model = create_model()
        self.chain = None

    def init_chain(self):
        if self.chain is None:
            template = """
                You are an expert at answering quetions about a time time table.
                You already have the records for the timetable:

                This record contains information about a class schedule including the unique identifier, day, slot, subject, subjectCode,
                subjectFullName, subjectType, subjectCredit, faculty, subjectDept, offeringDept,
                year, room, sem, code, session, mergedClass status, creation timestamp, update timestamp, version number, and degree.

                The subjectFullName, identified by its subjectCode, is being taught by the faculty assigned to the class. This subject, categorized under subjectType and carrying subjectCredit credits, belongs to the subjectDept and is offered by the offeringDept. 
                The class is scheduled on the specified day and slot, and it takes place in the room allocated for the sem. The session details, along with the year, provide the academic context, while the code uniquely identifies this class entry. Additional metadata such as mergedClass status, created_at timestamp, updated_at timestamp, version (__v), and degree help track the record’s administrative information.

                Here are some relevant records: {records}

                Here is the question to answer: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)
            self.chain = prompt | self.model

    def chat(self, question: str):
    
        self.init_chain()

        relevant_docs = get_documents(question)
        result = self.chain.invoke({"records":relevant_docs, "question":question})

        return result.content

rag = RAGPipeline()