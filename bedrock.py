import os
import boto3
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock




def bedrock_chain():
    bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",)

    llm = Bedrock(
    region_name="us-east-1",
    client= bedrock_runtime,
    model_id="anthropic.claude-v2",
    model_kwargs={"temperature": 0.0, "top_p": .2, "max_tokens_to_sample": 500})


    #warnings.filterwarnings('ignore')

    claude_prompt = PromptTemplate.from_template("""

    Human: You are an AI assistant for an Airline called WorldAir. You are helpful, polite friendly. Start by asking how you can help.
    Allowe users to ask questions relating to booking or WorldAir and avoid having conversations outside on traveling and booking business. It is very important that you finish collecting the following fields from the user. Ask two or three questions at the time:
    The user's full name that includes first name and family name.
    The user's departure location
    The user's destination location
    One way or return
    The user's dates of travel, check if the dates are flexible or non negotiable
    number of people traveling, ask both for adults and children under 12 years old
    The user's budget for the trip
    Any additional preferences or constraints the user may have

    Current conversation:
    <conversation_history>
    {history}
    </conversation_history>

    Here is the human's next reply:
    <human_reply>
    {input}
    </human_reply>

    Assistant:
    """)

    # PROMPT = PromptTemplate(
    #     input_variables=["history", "input"], template=claude_prompt
    # )

    #conversation.prompt = claude_prompt
    # turn verbose to true to see the full logs and documents
    #memory = ConversationBufferMemory(human_prefix="User", ai_prefix="Bot")


    conversation = ConversationChain(
        prompt=claude_prompt,
        llm=llm,
        verbose=True,
        memory=ConversationBufferMemory(),
    )
    
    return conversation



def run_chain(chain, prompt):
    num_tokens = chain.llm.get_num_tokens(prompt)
    return chain({"input": prompt}), num_tokens


def clear_memory(chain):
    return chain.memory.clear()