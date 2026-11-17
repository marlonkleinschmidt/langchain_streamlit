import os
import streamlit as st

from decouple import config

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI


os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

st.set_page_config(
    page_title='Estoque GPT',
    page_icon='📄',
)
st.header('Assistente de Estoque')

model_options = [
    'gpt-6.1-sol',   # recomendado: bom equilíbrio custo/precisão
    'gpt-6-luna',    # mais barato, bom para perguntas simples
    'gpt-6-astra',   # mais potente (e caro)
    'gpt-4o-mini',   # modelos antigos: mantenha só se ainda funcionarem na sua conta
    'gpt-4o',
]
selected_model = st.sidebar.selectbox(
    label='Selecione o modelo LLM',
    options=model_options,
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Este agente consulta um banco de dados de estoque utilizando um modelo GPT.')

st.write('Faça perguntas sobre o estoque de produtos, preços e reposições.')
user_question = st.text_input('O que deseja saber sobre o estoque?')


@st.cache_resource
def get_db():
    # Só as tabelas de negócio; as tabelas auth_* e django_* apenas confundem o modelo
    return SQLDatabase.from_uri(
        'sqlite:///estoque.db',
        include_tables=[
            'products_product',
            'brands_brand',
            'categories_category',
            'suppliers_supplier',
            'inflows_inflow',
            'outflows_outflow',
        ],
        sample_rows_in_table_info=3,
    )


db = get_db()

# Modelos GPT-6 raciocinam por padrão; com tools, isso só é aceito na Responses API
if selected_model.startswith('gpt-6'):
    model = ChatOpenAI(
        model=selected_model,
        use_responses_api=True,
    )
else:
    model = ChatOpenAI(model=selected_model)

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# As instruções que antes iam junto da pergunta agora ficam no prompt de sistema
prompt = ChatPromptTemplate.from_messages([
    ('system', '''
Você é um assistente que consulta um banco de dados SQLite de estoque.
Use as ferramentas disponíveis para responder perguntas sobre produtos,
preços, reposição de estoque e relatórios.

Regras:
- Antes de escrever uma query, verifique o schema das tabelas envolvidas.
- Escreva SQL puro, sem blocos de markdown.
- Nunca execute comandos que alterem dados (INSERT, UPDATE, DELETE, DROP).
- Dicas do schema: o nome do produto fica em products_product.title;
  saídas/vendas ficam em outflows_outflow e entradas/reposições em inflows_inflow.

A resposta final deve ter uma formatação amigável (markdown, listas ou tabelas)
e ser sempre em português brasileiro.
'''),
    ('human', '{input}'),
    MessagesPlaceholder('agent_scratchpad'),
])

agent = create_tool_calling_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15,
)

if st.button('Consultar'):
    if user_question:
        with st.spinner('Consultando o banco de dados...'):
            try:
                output = agent_executor.invoke({'input': user_question})
                resposta = output.get('output')
                # Na Responses API a saída pode vir como lista de blocos
                if isinstance(resposta, list):
                    resposta = ''.join(
                        bloco.get('text', '') if isinstance(bloco, dict) else str(bloco)
                        for bloco in resposta
                    )
                st.markdown(resposta)
            except Exception as e:
                st.error(f'Não foi possível concluir a consulta: {e}')
    else:
        st.warning('Por favor, insira uma pergunta.')