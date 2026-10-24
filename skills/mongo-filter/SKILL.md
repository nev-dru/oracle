---
name: mongo-filter
applies_to: [planner, check]
scope: tool:mongo
when: "Writing a JSON filter for mongo.find or mongo.count"
---
Mongo filters for `mongo.find` / `mongo.count`:
- A JSON object of equality matches: `{"email": "jane@example.com"}`, `{"account_id": 42}`.
  Numbers stay numbers; strings are quoted.
- Operators (`$where`, `$regex`, ...) are refused; use exact field values.
- Pick the collection by what is stored: people in `users`, companies in `customers`, purchases
  in `orders`. List collections first when unsure.
