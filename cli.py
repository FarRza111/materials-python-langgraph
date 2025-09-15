from graphs.email_agent import email_agent_graph
from example_emails import EMAILS


escalation_criteria = """"There's an immediate risk of electrical,
water, or fire damage"""

message_with_criteria = f"""
The escalation criteria is: {escalation_criteria}

Here's the email:
{EMAILS[3]}
"""
message_3 = {"messages": [("human", message_with_criteria)]}

for chunk in email_agent_graph.stream(message_3, stream_mode="values"):
    chunk["messages"][-1].pretty_print()



