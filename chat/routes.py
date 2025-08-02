from fastapi import APIRouter,Depends,Form
from auth.routes import authenticate
from chat.chat_query import answer_query


router=APIRouter()

@router.post("/chat")
async def chat(user=Depends(authenticate),message:str=Form(...)):
    return await answer_query(message,user["role"])