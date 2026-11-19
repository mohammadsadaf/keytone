import { trainModel } from './model.js';
self.onmessage = async event => {
  try {
    const result = await trainModel(event.data.sessions, event.data.options, progress => self.postMessage({ type: 'progress', ...progress }));
    self.postMessage({ type: 'done', ...result });
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};
