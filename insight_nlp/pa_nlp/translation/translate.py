#coding: utf8

from google.cloud import translate
import os
from pa_nlp import common as common

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = \
  common.get_module_path("Common") + \
  "/translation/ZSProject-94cb8e930aab.json"

def translate_sentence(text, target='Zh-cn'):
  # Imports the Google Cloud client library
  translate_client = translate.Client()
  translation = translate_client.translate(text, target_language=target)

  return translation['translatedText']

