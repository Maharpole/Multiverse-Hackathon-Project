CREATE TABLE IF NOT EXISTS Key_Associations (
    user_id TEXT,
    api_key TEXT,
    s3_object_key TEXT,
    PRIMARY KEY (api_key)
);