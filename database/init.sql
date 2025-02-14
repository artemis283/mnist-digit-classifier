-- database/init.sql
CREATE DATABASE mnist_db;

\c mnist_db;

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    predicted_digit INT NOT NULL,
    true_label INT
);
