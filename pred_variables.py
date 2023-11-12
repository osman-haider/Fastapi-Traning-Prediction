from pydantic import BaseModel

class variables_dtypes(BaseModel):
  N_Days: int
  Drug: int
  Age:int
  Sex: int
  Ascites: int
  Hepatomegaly: int
  Spiders: int
  Edema: int
  Bilirubin: float
  Cholesterol: float
  Albumin: float
  Copper: float
  Alk_Phos: float
  SGOT: float
  Tryglicerides: float
  Platelets: float
  Prothrombin: float
  Stage: float