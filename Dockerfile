# --- build the React frontend ---
FROM node:20-alpine AS web
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
# Leave empty: the API is served by the same container, so the browser uses relative /api URLs.
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# --- run the API + serve the built frontend ---
FROM node:20-alpine
ENV NODE_ENV=production DATA_DIR=/data FRONTEND_DIST=/app/frontend/dist NOTEBOOKS_DIR=/app/notebooks TRUST_PROXY=1 PORT=4000
WORKDIR /app/backend
COPY backend/package*.json ./
RUN npm ci --omit=dev
COPY backend ./
COPY --from=web /app/frontend/dist /app/frontend/dist
COPY notebooks /app/notebooks
RUN mkdir -p /data && chown -R node:node /data
VOLUME /data
USER node
EXPOSE 4000
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:4000/api/health || exit 1
CMD ["node", "src/server.js"]
