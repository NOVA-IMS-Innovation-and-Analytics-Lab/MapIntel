"""Implementation of the application."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import os
import warnings
from datetime import datetime
from textwrap import wrap
from typing import Any, cast

import numpy as np
from boto3 import Session as BotoSession
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.document_stores import OpenSearchDocumentStore
from haystack.nodes import AnswerParser, EmbeddingRetriever, PromptNode, PromptTemplate
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, rgb2hex
from nicegui import ui
from nicegui.elements.textarea import Textarea
from nicegui.events import GenericEventArguments, ValueChangeEventArguments
from opensearchpy import OpenSearch
from sagemaker.session import Session as SageMakerSession
from sagemaker.sklearn import SKLearnPredictor
from sklearn.preprocessing import LabelEncoder

from .config import CONFIG

warnings.filterwarnings('ignore')
load_dotenv()


def create_topics_mapping(unique_topics: list[str]) -> dict[str, str]:
    """Create mapping from topics to colors."""
    encoded_topics = LabelEncoder().fit_transform(unique_topics)
    norm = Normalize(vmin=encoded_topics.min(), vmax=encoded_topics.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=CONFIG['topics_mapping']['cmap'])
    rgba = mapper.to_rgba(encoded_topics)
    mapping = dict(zip(unique_topics, np.apply_along_axis(rgb2hex, 1, rgba), strict=True))
    return mapping


def create_document_store() -> OpenSearchDocumentStore:
    """Create an OpenSearch document store."""
    os_client = OpenSearch(
        f'https://{os.environ["OPENSEARCH_ENDPOINT"]}:{os.environ["OPENSEARCH_PORT"]}',
        http_auth=(os.environ['OPENSEARCH_USERNAME'], os.environ['OPENSEARCH_PASSWORD']),
    )
    index = os_client.indices.get(index='document')
    embedding_dim = index['document']['mappings']['properties']['embedding']['dimension']
    return OpenSearchDocumentStore(
        host=os.environ['OPENSEARCH_ENDPOINT'],
        port=os.environ['OPENSEARCH_PORT'],
        username=os.environ['OPENSEARCH_USERNAME'],
        password=os.environ['OPENSEARCH_PASSWORD'],
        embedding_dim=embedding_dim,
    )


def create_retriever(document_store: OpenSearchDocumentStore) -> EmbeddingRetriever:
    """Create the Haystack embedding retriever."""
    return EmbeddingRetriever(os.environ['HAYSTACK_RETRIEVER_MODEL'], document_store)


def create_dimensionality_reductioner() -> SKLearnPredictor:
    """Create the dimensionality reductioner."""
    sagemaker_session = SageMakerSession(boto_session=BotoSession(profile_name=os.environ['AWS_PROFILE_NAME']))
    dimensionality_reductioner = SKLearnPredictor(
        os.environ['SAGEMAKER_DIMENSIONALITY_REDUCTIONER_ENDPOINT'],
        sagemaker_session=sagemaker_session,
    )
    return dimensionality_reductioner


def create_qa_pipeline(retriever: EmbeddingRetriever) -> Pipeline:
    """Create the question answering pipeline."""
    reference_pattern = r"Document\[(\d+)\]"
    prompt_template = PromptTemplate(
        prompt=CONFIG['qa_pipeline']['prompt'],
        output_parser=AnswerParser(reference_pattern=reference_pattern),
    )
    qa_prompt = PromptNode(
        default_prompt_template=prompt_template,
        max_length=CONFIG['qa_pipeline']['max_length'],
        model_name_or_path=os.environ['SAGEMAKER_GENERATOR_MODEL_ENDPOINT'],
        model_kwargs={'aws_profile_name': os.environ['AWS_PROFILE_NAME']},
    )
    pipeline = Pipeline()
    pipeline.add_node(component=retriever, name='Retriever', inputs=['Query'])
    pipeline.add_node(component=qa_prompt, name='QAPromptNode', inputs=['Retriever'])
    return pipeline


def get_dates_range() -> tuple[int, int]:
    """Get the dates range."""
    os_client = OpenSearch(
        f'https://{os.environ["OPENSEARCH_ENDPOINT"]}:{os.environ["OPENSEARCH_PORT"]}',
        http_auth=(os.environ['OPENSEARCH_USERNAME'], os.environ['OPENSEARCH_PASSWORD']),
    )
    request_body = {
        'size': 0,
        'query': {'match_all': {}},
        'aggs': {'min': {'min': {'field': 'date'}}, 'max': {'max': {'field': 'date'}}},
    }
    response = os_client.search(body=request_body)
    dates_range = int(response['aggregations']['min']['value']), int(response['aggregations']['max']['value'])
    return dates_range


def get_unique_topics() -> list[str]:
    """Get all unique topics names."""
    all_topics = sorted({info['value'] for info in DOCUMENT_STORE.get_metadata_values_by_key('topic')})
    return all_topics


def update_filters_date(date_input: ValueChangeEventArguments) -> None:
    """Update the filters of date."""
    if isinstance(date_input.value, dict):
        filters['date'] = {
            '$gte': datetime.strptime(date_input.value['from'], '%Y-%m-%d').astimezone().toordinal(),
            '$lte': datetime.strptime(date_input.value['to'], '%Y-%m-%d').astimezone().toordinal(),
        }
    elif isinstance(date_input.value, str):
        filters['date'] = {
            '$gte': datetime.strptime(date_input.value, '%Y-%m-%d').astimezone().toordinal(),
            '$lte': datetime.strptime(date_input.value, '%Y-%m-%d').astimezone().toordinal(),
        }
    plot_map.refresh()
    plot_retrieved_documents.refresh()


def update_filters_topics(selection: GenericEventArguments) -> None:
    """Update the filters of topics."""
    filters['topic'] = {'$in': [arg['label'] for arg in selection.args]}
    plot_map.refresh()
    plot_retrieved_documents.refresh()


@ui.refreshable
def plot_map() -> None:
    """Plot the map of documents."""
    documents = DOCUMENT_STORE.get_all_documents(filters=filters)
    unique_topics = list({document.meta['topic'] for document in documents})
    topics_mapping = create_topics_mapping(unique_topics)
    data = []
    if 'embedding2d' in query:
        x, y = query['embedding2d']
        marker = {'size': 10, 'color': 'red', 'symbol': 'x-dot'}
        data.append(
            {
                'x': [x],
                'y': [y],
                'type': 'scatter',
                'name': 'query',
                'mode': 'markers',
                'marker': marker,
                'text': [query['value']],
                'hovertemplate': '<b>Query</b><br>%{text}',
            },
        )
    for topic in unique_topics:
        documents_topic = [document for document in documents if document.meta['topic'] == topic]
        trace = {
            'x': [document.meta['embedding2d'][0] for document in documents_topic],
            'y': [document.meta['embedding2d'][1] for document in documents_topic],
            'mode': 'markers',
            'type': 'scatter',
            'name': topic,
            'text': [
                '<br>'.join(
                    wrap(document.content, width=cast(int, CONFIG['plot_map']['wrap_width']))[
                        : cast(int, CONFIG['plot_map']['wrap_length'])
                    ],
                )
                for document in documents_topic
            ],
            'marker': {'size': 4, 'color': topics_mapping[topic], 'symbol': 'circle', 'opacity': 0.3},
            'hovertemplate': '<b>Content</b><br>%{text}...',
        }
        data.append(trace)
    ui.plotly({'data': data})


@ui.refreshable
def plot_retrieved_documents() -> None:
    """Plot the retrived documents."""
    if not query:
        return
    query['embedding'] = RETRIEVER.embed_queries(query['value']).reshape(-1)
    query['embedding2d'] = DIMENSIONALITY_REDUCTIONER.predict([query['embedding']]).reshape(-1)
    if not prompt_description['value']:
        qa_answer['value'] = QA_PIPELINE.run(query['value'], params={'Retriever': {'top_k': 1}})['answers'][0].answer
    else:
        qa_answer['value'] = cast(str, CONFIG['qa_answer']['default_value'])
    documents = DOCUMENT_STORE.query_by_embedding(query_emb=query['embedding'], filters=filters, top_k=top_k['value'])
    with ui.grid(columns=2):
        for number, document in enumerate(documents):
            with ui.card().tight():
                with ui.link('', document.meta['link']):
                    with ui.image(document.meta['media']) as image:
                        image.tailwind.height('48')
                        with ui.row():
                            ui.icon('tag', size='25px')
                            ui.label(f'{number + 1}').tailwind.font_size('xl').font_weight('bold')
                ui.separator().style('margin-bottom: 20px')
                with ui.row():
                    ui.icon('category', size='25px')
                    ui.label(f'{document.meta["topic"]}').tailwind.font_size('lg')
                with ui.row():
                    ui.icon('date_range', size='25px')
                    ui.label(f'{datetime.fromordinal(document.meta["date"]).date()}').tailwind.font_size('lg')
                ui.separator().style('margin-top: 20px')
                with ui.card_section():
                    ui.label(document.content)


@ui.refreshable
def plot_prompt() -> None:
    """Plot the prompt."""
    with ui.row().style('margin-top: 50px') as row:
        row.tailwind.align_items('center').align_self('center')
        with ui.column():
            textarea_description = ui.textarea(
                placeholder=CONFIG['plot_prompt']['placeholder_tokens']
                if prompt_description['value']
                else CONFIG['plot_prompt']['placeholder_question'],
            )
            ui.button('Submit', on_click=lambda: update_query(textarea_description))


@ui.refreshable
def plot_answer() -> None:
    """Plot the answer to the question."""
    msg = '\n'.join(wrap(qa_answer['value'], width=80))
    ui.chat_message(msg, name='Robot', avatar='https://robohash.org/ui')


def update_top_k(slider: GenericEventArguments) -> None:
    """Update the top k slider."""
    top_k['value'] = slider.args
    plot_retrieved_documents.refresh()


def update_query(textarea_query: Textarea) -> None:
    """Update the query."""
    query['value'] = textarea_query.value
    plot_retrieved_documents.refresh()
    plot_answer.refresh()
    plot_map.refresh()


def update_prompt_kind() -> None:
    """Update the prompt kind."""
    qa_answer['value'] = cast(str, CONFIG['qa_answer']['default_value'])
    prompt_description['value'] = not prompt_description['value']
    plot_prompt.refresh()
    plot_answer.refresh()


filters: dict[str, Any] = {}
top_k: dict[str, int] = {'value': cast(int, CONFIG['top_k']['default_value'])}
query: dict[str, Any] = {}
prompt_description: dict[str, bool] = {'value': True}
qa_answer: dict[str, str] = {'value': cast(str, CONFIG['qa_answer']['default_value'])}
DOCUMENT_STORE = create_document_store()
RETRIEVER = create_retriever(DOCUMENT_STORE)
DIMENSIONALITY_REDUCTIONER = create_dimensionality_reductioner()
QA_PIPELINE = create_qa_pipeline(RETRIEVER)


def main() -> None:  # noqa: PLR0915
    """Main application."""

    # Get dates range
    (
        min_date,
        max_date,
    ) = get_dates_range()

    # Get all topics
    all_topics = get_unique_topics()

    # Left drawer
    with ui.left_drawer(bordered=True, elevated=True).style('background-color: #d9d9d9') as left_drawer:
        left_drawer.props('width=600')

        with ui.column():
            ui.markdown('#### Options').tailwind.align_self('center')

            ui.separator()

            # Date
            min_date_dt, max_date_dt = datetime.fromordinal(min_date), datetime.fromordinal(max_date)
            with ui.row().style('margin-top: 50px') as row:
                row.tailwind.align_items('center')
                with ui.column() as col:
                    col.tailwind.align_items('center')
                    ui.icon('date_range', size='100px')
                    ui.label('Dates range').tailwind.font_size('xs')
                ui.date({'from': min_date_dt, 'to': max_date_dt}, on_change=update_filters_date).props('range').props(
                    f'default-year-month={min_date_dt.strftime("%Y/%m")} :options="date '
                    f'=> date >= \'{min_date_dt.strftime("%Y/%m/%d")}\' &&  '
                    f'date <= \'{max_date_dt.strftime("%Y/%m/%d")}\'"',
                ).props('landscape')

            # Topics
            with ui.row().style('margin-top: 50px') as row:
                row.tailwind.align_items('center')
                with ui.column() as col:
                    col.tailwind.align_items('center')
                    ui.icon('category', size='100px')
                    ui.label('Selected topics').tailwind.font_size('xs')
                ui.select(all_topics, multiple=True, value=all_topics).props('use-chips').on(
                    'update:model-value',
                    update_filters_topics,
                    throttle=5.0,
                    leading_events=False,
                ).tailwind.width('2/3')

            # Number of retrived documents
            with ui.row().style('margin-top: 50px') as row:
                row.tailwind.align_items('center')
                with ui.column() as col:
                    col.tailwind.align_items('center')
                    ui.icon('tag', size='100px')
                    ui.label('Retrieved documents').tailwind.font_size('xs')
                ui.knob(min=1, max=30, step=1, show_value=True, size='120px', value=top_k['value']).props(
                    'label-always',
                ).on('update:model-value', update_top_k, throttle=2.0, leading_events=False)

    # Right drawer
    with ui.right_drawer(bordered=True, elevated=True).style('background-color: #d9d9d9') as right_drawer:
        right_drawer.props('width=300')

        with ui.column():
            ui.markdown('#### Prompts').tailwind.align_self('center')

            ui.separator()

            # Prompt
            with ui.row().style('margin-top: 50px') as row:
                row.tailwind.align_items('center').align_self('center')
                ui.toggle(['Description', 'Question'], value='Description', on_change=update_prompt_kind)
            plot_prompt()

    # Header
    with ui.header(bordered=True, elevated=True).classes('bg-primary'):
        with ui.element('q-toolbar'):
            ui.button(on_click=lambda: left_drawer.toggle(), icon='menu').props('dense flat round color=white')
            with ui.element('q-toolbar-title').props('align=center'):
                ui.markdown('### Documents Map Intelligence')
            ui.button(on_click=lambda: right_drawer.toggle(), icon='menu').props('dense flat round color=white')

    # Map
    with ui.row() as row:
        row.tailwind.align_items('center')
        ui.icon('travel_explore', size='80px')
        ui.label('Map').tailwind.font_size('2xl').font_weight('bold')
    with ui.row():
        plot_map()
        plot_answer()

    # Similar documents
    with ui.row() as row:
        row.tailwind.align_items('center')
        ui.icon('insights', size='80px')
        ui.label('Intel').tailwind.font_size('2xl').font_weight('bold')
    plot_retrieved_documents()

    ui.run()


main()
