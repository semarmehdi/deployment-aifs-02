import uvicorn
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import Literal, List, Union
import mlflow
import os
from dotenv import load_dotenv

load_dotenv(override=True)
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

REGISTERED_MODEL_NAME = "ibm_attrition_detector"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"

MODEL = mlflow.sklearn.load_model(MODEL_URI)


class BlogArticles(BaseModel):
    title: str = "default title"
    content: str
    author: str = "Anonymous Author"


class IBMemployee(BaseModel):
    Age: Union[int, float]
    BusinessTravel: str
    DailyRate: Union[int, float]
    Department: str
    DistanceFromHome: Union[int, float]
    Education: Union[int, float]
    EducationField: str
    EmployeeCount: Union[int, float]
    EmployeeNumber: Union[int, float]
    EnvironmentSatisfaction: Union[int, float]
    Gender: str
    HourlyRate: Union[int, float]
    JobInvolvement: Union[int, float]
    JobLevel: Union[int, float]
    JobRole: str
    JobSatisfaction: Union[int, float]
    MaritalStatus: str
    MonthlyIncome: Union[int, float]
    MonthlyRate: Union[int, float]
    NumCompaniesWorked: Union[int, float]
    Over18: str
    OverTime: str
    PercentSalaryHike: Union[int, float]
    PerformanceRating: Union[int, float]
    RelationshipSatisfaction: Union[int, float]
    StandardHours: Union[int, float]
    StockOptionLevel: Union[int, float]
    TotalWorkingYears: Union[int, float]
    TrainingTimesLastYear: Union[int, float]
    WorkLifeBalance: Union[int, float]
    YearsAtCompany: Union[int, float]
    YearsInCurrentRole: Union[int, float]
    YearsSinceLastPromotion: Union[int, float]
    YearsWithCurrManager: Union[int, float]


###
# Here you can define some configurations
###

app = FastAPI()

###
# Here you define enpoints
###


@app.get("/")
async def index():

    message = "Hello aifs 02 comment allez vous ?! That `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"

    return message


@app.get("/hello")
async def hi():
    return "Hello there 🤗"


@app.get("/custom-greetings")
async def custom_greetings(name: str = "Mr (or Miss) Nobody"):
    greetings = {"Message": f"Hello {name} How are you today?"}
    return greetings


@app.get("/blog-articles/{blog_id}")
async def read_blog_article(blog_id: int, params_test: str):

    articles = pd.read_csv(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv"
    )
    if blog_id > len(articles):
        response = {"msg": "We don't have that many articles!"}
    else:
        article = articles.iloc[blog_id, :]
        response = {
            "title": article.title,
            "content": article.content,
            "author": article.author,
            "params_test": params_test,
        }

    return response


@app.post("/create-blog-article")
async def create_blog_article(blog_article: dict):
    df = pd.read_csv(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv"
    )
    new_article = pd.DataFrame(
        {
            "id": len(df) + 1,
            "title": blog_article.title,
            "content": blog_article.content,
            "author": blog_article.author,
        }
    )

    df = pd.concat([df, new_article])

    return df.to_json()


@app.post("/create-blog-article-dict")
async def create_blog_article(blog_article: dict):
    df = pd.read_csv(
        "https://full-stack-assets.s3.eu-west-3.amazonaws.com/Deployment/articles.csv"
    )
    new_article = pd.DataFrame(
        [
            {
                "id": len(df) + 1,
                "title": blog_article["title"],
                "content": blog_article["content"],
                "author": blog_article["author"],
            }
        ]
    )

    df = pd.concat([df, new_article], ignore_index=True)

    return df.to_json()


@app.post("/another-post-endpoint")
async def another_post_endpoint(blog_article: BlogArticles):
    example_data = {
        "title": blog_article.title,
        "content": blog_article.content,
        "author": blog_article.author,
    }
    return example_data


@app.post("/predict-attrition")
async def predict_attrition(employee_data: IBMemployee):
    # Objet Pydantic -> dict -> DataFrame d'une ligne (noms de colonnes conservés)
    df = pd.DataFrame([employee_data.model_dump()])  # .dict() si Pydantic v1

    prediction = MODEL.predict(df)

    # numpy n'est pas sérialisable en JSON -> on convertit en type Python
    return {"prediction": prediction.tolist()[0]}


# @app.get("/get-data")
# async def get_data(ligne, colonne):
#     df = pd.read_sql()
#     return df.iloc[ligne, colonne]


# @app.post("/predict")
# async def predict(data):
#     return model.predict(data)
