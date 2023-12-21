# VRAM Calculator

An app which helps to estimate GPU VRAM needed for training/inference transformer runs. Check it out here: [vram.asmirnov.xyz](https://vram.asmirnov.xyz/).

## Details

You can find main logic of calculating the result in [./app/\_lib/index.ts](./app/_lib/index.ts).

## Development

```bash
npm install
npm run dev
```

## Deployment

```bash
npm install
npm run build
```

It will result with `./out` folder which can be served with any webserver (e.g. nginx).
