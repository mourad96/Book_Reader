import logging
import requests

class TextProcessor:
    def __init__(self, api_key, external_user_id):
        self.api_key = api_key
        self.external_user_id = external_user_id

    def create_session(self):
        """Create a chat session."""
        create_session_url = 'https://api.on-demand.io/chat/v1/sessions'
        create_session_headers = {'apikey': self.api_key}
        create_session_body = {
            "pluginIds": [],
            "externalUserId": self.external_user_id
        }

        response = requests.post(create_session_url, headers=create_session_headers, json=create_session_body)
        if response.status_code != 201:
            raise Exception(f"Failed to create session: {response.text}")
        return response.json()['data']['id']

    def submit_query(self, session_id, text):
        """Submit a query to the session."""
        submit_query_url = f'https://api.on-demand.io/chat/v1/sessions/{session_id}/query'
        submit_query_headers = {'apikey': self.api_key}
        submit_query_body = {
            "endpointId": "predefined-openai-gpt4o",
            "query": f"Please summarize the following text:\n{text}",
            "pluginIds": ["plugin-1712327325", "plugin-1713962163"],
            "responseMode": "sync"
        }

        response = requests.post(submit_query_url, headers=submit_query_headers, json=submit_query_body)
        if response.status_code != 200:
            raise Exception(f"Failed to submit query: {response.text}")
        return response.json()['data']['answer']

    def summarize_text(self, text):
        """Summarize text using the external API."""
        try:
            logging.info("Creating chat session for summarization...")
            session_id = self.create_session()
            logging.info(f"Chat session created with ID: {session_id}")
            
            logging.info("Submitting text to the external API for summarization...")
            summary = self.submit_query(session_id, text)
            return summary
        except Exception as e:
            logging.error(f"Failed to summarize text: {e}")
            return text