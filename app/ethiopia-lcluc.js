///////////////////////////////////////////////////////////////
// Ethiopia / Amhara Land Cover Explorer
///////////////////////////////////////////////////////////////

// -------------------------
// 1) Inputs
// -------------------------
var ethiopiaBoundary = ee.FeatureCollection(
  'projects/ee-jacaraba-ethiopia/assets/boundaries/Amhara_Study_Area_Boundary_4buf10km_EPSG_GEE'
);

var worldCov10m = ee.ImageCollection('ESA/WorldCover/v100')
  .first()
  .clip(ethiopiaBoundary);

var gladCropland = ee.ImageCollection('users/potapovpeter/Global_cropland_2019')
  .map(function(img) {
    var clipped = img.clip(ethiopiaBoundary);
    return clipped.updateMask(clipped.gt(0));
  });

var canopyHeight = ee.ImageCollection(
  'projects/sat-io/open-datasets/facebook/meta-canopy-height'
).mosaic().clip(ethiopiaBoundary);


// GSFC DSG assets
var lc_2009_2016 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2009_2016_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var lc_2018_2022 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2018_2022_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var lc_2017_2024 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_LCLU_5class_2017_2024_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2009_2016 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2009_2016_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2018_2022 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2018_2022_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);

var nobs_2017_2024 = ee.Image(
  'projects/gsfc-dsg/assets/Amhara_nobservations_2017_2024_2m_native_cog_clean_cog'
).clip(ethiopiaBoundary);


// -------------------------
// 2) Visualization
// -------------------------
var palettes = require('users/gena/packages:palettes');

var lcPalette = [
  '#ffaa00', // Crop
  '#267300', // Tree/Shrub
  '#ffffbe', // Grass
  '#730000', // Built
  '#0070ff'  // Water
];

var lcNames = [
  'Crop',
  'Tree / Shrub',
  'Grass',
  'Built',
  'Water'
];

var worldCovPalette = [
  '#006400',
  '#ffbb22',
  '#ffff4c',
  '#f096ff',
  '#fa0000',
  '#b4b4b4',
  '#f0f0f0',
  '#0064c8',
  '#0096a0',
  '#00cf75',
  '#fae6a0'
];

var worldCovNames = [
  '10 Trees',
  '20 Shrubland',
  '30 Grassland',
  '40 Cropland',
  '50 Built-up',
  '60 Barren / sparse vegetation',
  '70 Snow and ice',
  '80 Open water',
  '90 Herbaceous wetland',
  '95 Mangroves',
  '100 Moss and lichen'
];

var lcVis = {
  min: 0,
  max: 4,
  palette: lcPalette
};

var nobsVis = {
  min: 0,
  max: 100,
  palette: palettes.matplotlib.magma[7]
};

var chmVis = {
  min: 0,
  max: 20,
  palette: palettes.matplotlib.viridis[7]
};


// -------------------------
// 3) Map setup
// -------------------------
Map.setOptions('SATELLITE');
Map.centerObject(ethiopiaBoundary, 7);
Map.style().set('cursor', 'crosshair');

var boundaryLayer = ui.Map.Layer(
  ethiopiaBoundary.style({
    color: 'ffffff',
    fillColor: '00000000',
    width: 2
  }),
  {},
  'Amhara Study Boundary',
  true
);

Map.add(boundaryLayer);


// -------------------------
// 4) Add map layers
// -------------------------
function addLayer(image, vis, name, shown, opacity) {
  var layer = ui.Map.Layer(image, vis, name, shown, opacity || 1.0);
  Map.add(layer);
  return layer;
}

var layers = {
  worldCov: addLayer(
    worldCov10m,
    {bands: ['Map']},
    'ESA WorldCover 10 m',
    false
  ),

  glad: addLayer(
    gladCropland,
    {palette: ['#FFA500'], min: 0, max: 1},
    'GLAD Cropland 2019',
    false
  ),

  chm: addLayer(
    canopyHeight,
    chmVis,
    'Meta Canopy Height 1 m',
    false
  ),

  lc0916: addLayer(
    lc_2009_2016,
    lcVis,
    'GSFC LCLU 2 m: 2009–2016',
    false
  ),

  lc1822: addLayer(
    lc_2018_2022,
    lcVis,
    'GSFC LCLU 2 m: 2018–2022',
    false
  ),

  lc1724: addLayer(
    lc_2017_2024,
    lcVis,
    'GSFC LCLU 2 m: 2017–2024',
    true
  ),

  nobs0916: addLayer(
    nobs_2009_2016,
    nobsVis,
    'Observations: 2009–2016',
    false,
    0.85
  ),

  nobs1822: addLayer(
    nobs_2018_2022,
    nobsVis,
    'Observations: 2018–2022',
    false,
    0.85
  ),

  nobs1724: addLayer(
    nobs_2017_2024,
    nobsVis,
    'Observations: 2017–2024',
    false,
    0.85
  )
};


// -------------------------
// 5) UI helpers
// -------------------------
function titleLabel(text) {
  return ui.Label(text, {
    fontSize: '22px',
    fontWeight: 'bold',
    color: '#4A997E',
    margin: '8px 8px 4px 8px'
  });
}

function subtitleLabel(text) {
  return ui.Label(text, {
    fontSize: '13px',
    color: '#555555',
    margin: '0 8px 12px 8px'
  });
}

function sectionLabel(text) {
  return ui.Label(text, {
    fontWeight: 'bold',
    fontSize: '15px',
    color: '#2F6F5E',
    margin: '14px 8px 6px 8px'
  });
}

function makeCheckbox(label, layer, defaultValue) {
  var checkbox = ui.Checkbox({
    label: label,
    value: defaultValue,
    style: {margin: '2px 8px'}
  });

  checkbox.onChange(function(checked) {
    layer.setShown(checked);
  });

  return checkbox;
}

function makeOpacitySlider(layer, label, defaultValue) {
  var slider = ui.Slider({
    min: 0,
    max: 1,
    value: defaultValue,
    step: 0.05,
    style: {
      stretch: 'horizontal',
      margin: '0 8px 8px 8px'
    }
  });

  slider.onChange(function(value) {
    layer.setOpacity(value);
  });

  return ui.Panel([
    ui.Label(label, {
      fontSize: '11px',
      color: '#666666',
      margin: '4px 8px 0 8px'
    }),
    slider
  ]);
}

function makeLegendRow(color, name) {
  return ui.Panel({
    widgets: [
      ui.Label({
        style: {
          backgroundColor: color,
          padding: '8px',
          margin: '0 0 4px 0'
        }
      }),
      ui.Label({
        value: name,
        style: {
          margin: '0 0 4px 6px',
          fontSize: '12px'
        }
      })
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

function makeDiscreteLegend(title, colors, names) {
  var legend = ui.Panel({
    style: {
      margin: '8px',
      padding: '8px',
      backgroundColor: 'rgba(255,255,255,0.90)',
      border: '1px solid #dddddd'
    }
  });

  legend.add(ui.Label(title, {
    fontWeight: 'bold',
    fontSize: '13px',
    color: '#333333',
    margin: '0 0 6px 0'
  }));

  for (var i = 0; i < colors.length; i++) {
    legend.add(makeLegendRow(colors[i], names[i]));
  }

  return legend;
}

function makeGradientLegend(title, vis, leftLabel, rightLabel) {
  var lon = ee.Image.pixelLonLat().select('longitude');
  var gradient = lon
    .multiply((vis.max - vis.min) / 100.0)
    .add(vis.min);

  var legendImage = gradient.visualize(vis);

  var thumb = ui.Thumbnail({
    image: legendImage,
    params: {
      bbox: '0,0,100,8',
      dimensions: '240x18'
    },
    style: {
      stretch: 'horizontal',
      margin: '4px 0'
    }
  });

  var labels = ui.Panel({
    widgets: [
      ui.Label(leftLabel, {fontSize: '11px'}),
      ui.Label('', {stretch: 'horizontal'}),
      ui.Label(rightLabel, {fontSize: '11px'})
    ],
    layout: ui.Panel.Layout.Flow('horizontal')
  });

  return ui.Panel({
    widgets: [
      ui.Label(title, {
        fontWeight: 'bold',
        fontSize: '13px',
        color: '#333333'
      }),
      thumb,
      labels
    ],
    style: {
      margin: '8px',
      padding: '8px',
      backgroundColor: 'rgba(255,255,255,0.90)',
      border: '1px solid #dddddd'
    }
  });
}


// -------------------------
// 6) Main control panel
// -------------------------
var panel = ui.Panel({
  style: {
    position: 'top-right',
    width: '370px',
    maxHeight: '95%',
    padding: '10px',
    backgroundColor: 'rgba(255,255,255,0.93)'
  }
});

panel.add(titleLabel('Amhara Land Cover Explorer'));

panel.add(subtitleLabel(
  'Compare 2 m land-cover products, cropland extent, canopy height, and observation density across Amhara, Ethiopia.'
));

panel.add(sectionLabel('Reference Layers'));
panel.add(makeCheckbox('Amhara Study Boundary', boundaryLayer, true));
panel.add(makeCheckbox('ESA WorldCover 10 m', layers.worldCov, false));
panel.add(makeCheckbox('GLAD Cropland 2019', layers.glad, false));
panel.add(makeCheckbox('Meta Canopy Height 1 m', layers.chm, false));

panel.add(sectionLabel('Amhara LCLU 2 m Products'));
panel.add(makeCheckbox('LCLU 2009–2016', layers.lc0916, false));
panel.add(makeCheckbox('LCLU 2018–2022', layers.lc1822, false));
panel.add(makeCheckbox('LCLU 2017–2024', layers.lc1724, true));
panel.add(makeOpacitySlider(layers.lc1724, 'Opacity: LCLU 2017–2024', 1.0));

panel.add(sectionLabel('Observation Density Layers'));
panel.add(makeCheckbox('Observations 2009–2016', layers.nobs0916, false));
panel.add(makeCheckbox('Observations 2018–2022', layers.nobs1822, false));
panel.add(makeCheckbox('Observations 2017–2024', layers.nobs1724, false));
panel.add(makeOpacitySlider(layers.nobs1724, 'Opacity: Observations 2017–2024', 0.85));

panel.add(sectionLabel('Legends'));
panel.add(makeDiscreteLegend('ESA WorldCover', worldCovPalette, worldCovNames));
panel.add(makeGradientLegend('Observation Count', nobsVis, '0', '100+'));
panel.add(makeGradientLegend('Canopy Height', chmVis, '0 m', '20 m'));


// -------------------------
// 7) Final app layout
// -------------------------
// Important:
// Do NOT use ui.root.clear()
// Do NOT use ui.root.add(Map)
// Do NOT use ui.SplitPanel()
// Do NOT use ui.root.insert(1, panel)

Map.add(panel);


// -------------------------
// 8) Map badge
// -------------------------
var mapBadge = ui.Label('GSFC DSG | Amhara 2 m LCLU Explorer', {
  position: 'bottom-right',
  padding: '6px 10px',
  backgroundColor: 'rgba(0,0,0,0.55)',
  color: 'white',
  fontSize: '12px',
  fontWeight: 'bold'
});

Map.add(mapBadge);
