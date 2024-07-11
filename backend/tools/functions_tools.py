import json
from datetime import datetime
from langchain.tools import Tool
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Any, Dict
from pydantic import BaseModel
from functools import lru_cache
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Union

logger = logging.getLogger(__name__)

class ChartData(BaseModel):
    type: str
    data: str

class TableData(BaseModel):
    headers: List[str]
    rows: List[List[Any]]

class ChartInput(BaseModel):
    columns: List[str]
    data: List[List[Any]]
    chart_type: str

@lru_cache(maxsize=100)
def run_query_save_results(db: Session, query: str, params: dict = None) -> List[Dict[str, Any]]:
    try:
        result = db.execute(text(query), params)
        return [dict(row) for row in result]
    except Exception as e:
        logger.error(f"Database query error: {str(e)}")
        raise


@lru_cache(maxsize=1)
def get_columns_descriptions() -> str:
    from tools.tools_constants import COLUMNS_DESCRIPTIONS
    return json.dumps(COLUMNS_DESCRIPTIONS, ensure_ascii=False)

def get_today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def generate_chart(input_data: Union[str, dict, ChartInput] = None, **kwargs):
    try:
        if input_data is None and kwargs:
            # If input_data is not provided, use kwargs
            data = kwargs
        elif isinstance(input_data, str):
            # If input is a string, try to parse it as JSON
            data = json.loads(input_data)
        elif isinstance(input_data, dict):
            # If input is already a dictionary, use it directly
            data = input_data
        elif isinstance(input_data, ChartInput):
            # If input is a ChartInput object, convert it to a dict
            data = input_data.dict()
        else:
            raise ValueError("Invalid input type for generate_chart")

        # Ensure all required fields are present
        required_fields = ['columns', 'data', 'chart_type']
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        columns = data['columns']
        chart_data = data['data']
        chart_type = data['chart_type']
        
        return {
            'columns': columns,
            'data': chart_data,
            'chart_type': chart_type
        }

    except Exception as e:
        logger.error(f"Error in generate_chart: {str(e)}", exc_info=True)
        return f"Error preparing chart data: {str(e)}"
        
def sql_agent_tools(db: Session):
    tools = [
        
        Tool.from_function(
            func=lambda _: get_columns_descriptions(),
            name="get_columns_descriptions",
            description="Useful for getting the descriptions of columns in the table.",
        ),
        Tool.from_function(
            func=lambda _: get_today_date(),
            name="get_today_date",
            description="Useful for getting today's date.",
        ),
        
        StructuredTool.from_function(
            func=generate_chart,
            name="generate_chart",
            description="Prepares data for chart generation. Input: JSON, dict, or args for columns, data, and chart_type. Returns: Dict with processed data for later chart creation.",
            args_schema=ChartInput
        ),
    ]
    return tools

import matplotlib.pyplot as plt
import io
import base64

def create_chart_image(chart_data):
    try:
        columns = chart_data['columns']
        data = chart_data['data']
        chart_type = chart_data['chart_type']

        # Create a DataFrame
        df = pd.DataFrame(data, columns=columns)

        plt.figure(figsize=(10, 6))
        if chart_type == "bar":
            df.plot(kind="bar", x=columns[0], y=columns[1])
        elif chart_type == "line":
            df.plot(kind="line", x=columns[0], y=columns[1])
        elif chart_type == "pie":
            plt.pie(df[columns[1]], labels=df[columns[0]], autopct='%1.1f%%')
        elif chart_type == "scatter":
            plt.scatter(df[columns[0]], df[columns[1]])
        
        plt.title(f"{chart_type.capitalize()} Chart")
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()  # Close the figure to free up memory

        return f"![{chart_type.capitalize()} Chart](data:image/png;base64,{image_base64})"
    except Exception as e:
        return f"Error creating chart: {str(e)}"