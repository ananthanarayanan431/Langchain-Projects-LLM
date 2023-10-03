
import os
import streamlit as st
import replicate


from constant import REPLICATE_API_TOKEN


st.set_page_config("LLAMA-2 ChatBot with Streamlit")

with st.sidebar:
    st.title("LLama 2 ChatBOT")
    st.header("Parameters")

    replicate_key = st.text_input("Enter the Replicate API key : ",type="password")

    if not (replicate_key.startswith('r8_') and len(replicate_key)==40):
        st.warning("Please enter a valid API key") #icon=""
    else:
        st.success("Proceed to Entering your Prompt!")

    st.subheader("Models and Parameter")

    model = st.selectbox("Choose a Different LLAMA Model!",['Llama 2 7B','Llama 2 13B','Llama 2 70B'],key="select_model")


    if model=='Llama 2 7B':
        llm="meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    elif model=="Llama 2 13B":
        llm="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
    else:
        llm="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

    temperature = st.slider("temperature",min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.slider("top_p",min_value=0.01, max_value=1.0, value=0.9, step=0.01)

    max_length = st.slider("Max_Length",min_value=64, max_value=4096, value=512, step=8)

    st.markdown("I make content on AI updates and recent technologies [Linkedin](https://www.linkedin.com/in/rananthanarayananofficial/)")


os.environ["REPLICATE_API_TOKEN"] = replicate_key

if "messages" not in st.session_state.keys():
    st.session_state.message = [{"role":"assistant","content":"How may I assist you today?"}]

for mesage in st.session_state.message:
    with st.chat_message(mesage['role']):
        st.write(mesage['content'])

def clear_chat_history():
    st.session_state.message=[{'role':'assistant','content':'How May I assist you Today?'}]

st.sidebar.button('Clear Chat History',on_click=clear_chat_history)

def gnerate_llama2_response(prompt_input):
    deafult_system_prompt = "you are helfull assistant. You do not reponse as 'user' or pretend to be 'user'. you only respond once as assistant"
    for data in st.session_state.message:
        if data['role']=='user':
            deafult_system_prompt+= "User: " + data['content'] + '\n\n'
        else:
            deafult_system_prompt += "Assistant" + data['content'] + '\n\n'
    output = replicate.run(llm,
                           input={'prompt':f"{deafult_system_prompt} {prompt_input} Assistant: ",
                                "temperature":temperature,
                                  "top_p":top_p,
                                  "max_length":max_length,
                                  "repition_penality":1})

    return output

if prompt:= st.chat_input(disabled=not replicate_key):
    st.session_state.message.append({'role':'user','content':prompt})
    with st.chat_message('user'):
        st.write(prompt)

if st.session_state.message[-1]['role']!='assistant':
    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = gnerate_llama2_response(prompt)
            placeholder = st.empty()
            full_response=' '
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)

    mesage = {'role':'assistant','content':full_response}
    st.session_state.message.append(mesage)
