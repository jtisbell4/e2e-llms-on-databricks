from databricks.sdk import WorkspaceClient
from databricks.sdk.core import DatabricksError
import dotenv
w = WorkspaceClient(auth_type='pat')

scope_name = 'llama2-chat-demo'

try:
    w.secrets.create_scope(scope=scope_name)
except DatabricksError:
    pass

secrets = dotenv.dotenv_values('.env')

for s in secrets:
    w.secrets.put_secret(scope=scope_name, key=s, string_value=secrets[s])
