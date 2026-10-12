import React, { useState, useEffect } from 'react';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { UploadZone } from './components/UploadZone';
import { MriViewer } from './components/MriViewer';
import { SegmentationPanel } from './components/SegmentationPanel';
import { VolumeTable } from './components/VolumeTable';
import { QualityControlPanel } from './components/QualityControlPanel';
import { EvaluationPanel } from './components/EvaluationPanel';
import { ReportsView } from './components/ReportsView';
import { ModelsView } from './components/ModelsView';
import { AboutLimitations } from './components/AboutLimitations';

import { 
  ScanSessionData, 
  VolumeItem, 
  SegmentationJobResult, 
  EvaluationResult, 
  ModelGuide 
} from './types';
import { 
  uploadScan, 
  loadSampleScan, 
  runSegmentation, 
  evaluateSampleReference, 
  evaluateUploadedReference, 
  fetchModels, 
  deleteScanSession 
} from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [scanData, setScanData] = useState<ScanSessionData | null>(null);
  const [volumes, setVolumes] = useState<VolumeItem[]>([]);
  const [currentJob, setCurrentJob] = useState<SegmentationJobResult | null>(null);
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [models, setModels] = useState<ModelGuide[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch available models on load
  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data))
      .catch((err) => console.warn('Could not load models list:', err));
  }, []);

  const handleUpload = async (file: File, alias: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await uploadScan(file, alias);
      setScanData(data);
      setVolumes([]);
      setCurrentJob(null);
      setEvaluation(null);
      setActiveTab('viewer');
    } catch (err: any) {
      setError(err.message || 'File upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await loadSampleScan();
      setScanData(data);
      setVolumes([]);
      setCurrentJob(null);
      setEvaluation(null);
      setActiveTab('viewer');
    } catch (err: any) {
      setError(err.message || 'Failed to load sample scan');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSegmentation = async (adapterId: string) => {
    if (!scanData) return;
    setLoading(true);
    setError(null);
    try {
      const result = await runSegmentation(scanData.scan_id, adapterId);
      setCurrentJob(result);
      if (result.volumes) {
        setVolumes(result.volumes);
      }
      if (result.qc_report) {
        setScanData((prev) => prev ? { ...prev, qc_report: result.qc_report! } : prev);
      }
      if (result.status === 'Completed' || result.status === 'Demo mode') {
        setActiveTab('viewer');
      }
    } catch (err: any) {
      setError(err.message || 'Segmentation execution error');
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluateSample = async () => {
    if (!scanData) return;
    setLoading(true);
    setError(null);
    try {
      const res = await evaluateSampleReference(scanData.scan_id);
      setEvaluation(res);
    } catch (err: any) {
      setError(err.message || 'Sample reference evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleEvaluateUpload = async (file: File) => {
    if (!scanData) return;
    setLoading(true);
    setError(null);
    try {
      const res = await evaluateUploadedReference(scanData.scan_id, file);
      setEvaluation(res);
    } catch (err: any) {
      setError(err.message || 'Uploaded reference evaluation failed');
    } finally {
      setLoading(false);
    }
  };

  const handleClearSession = async () => {
    if (scanData) {
      try {
        await deleteScanSession(scanData.scan_id);
      } catch (err) {
        console.warn('Session cleanup:', err);
      }
    }
    setScanData(null);
    setVolumes([]);
    setCurrentJob(null);
    setEvaluation(null);
    setActiveTab('dashboard');
  };

  const isDemoActive = Boolean(currentJob?.is_demo || scanData?.alias.toLowerCase().includes('synthetic'));

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <DisclaimerBanner isDemo={isDemoActive} />
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        hasScan={Boolean(scanData)}
        scanAlias={scanData?.alias}
        isDemo={isDemoActive}
        onClearSession={handleClearSession}
      />

      <main style={{ flex: 1, maxWidth: '1280px', width: '100%', margin: '0 auto', padding: '28px 24px' }}>
        {activeTab === 'dashboard' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
            <Dashboard
              hasScan={Boolean(scanData)}
              scanData={scanData}
              models={models}
              onLoadSample={handleLoadSample}
              onNavigateToViewer={() => setActiveTab('viewer')}
              onNavigateToUpload={() => {}}
              loading={loading}
            />
            <div className="glass-panel" style={{ padding: '28px' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffffff', marginBottom: '6px' }}>
                Load Research MRI Scan
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '18px' }}>
                Upload an existing 3D NIfTI file or load the built-in synthetic T1-weighted benchmark scan.
              </p>
              <UploadZone
                onUpload={handleUpload}
                onLoadSample={handleLoadSample}
                loading={loading}
                error={error}
              />
            </div>
          </div>
        )}

        {activeTab === 'viewer' && scanData && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <SegmentationPanel
              scanId={scanData.scan_id}
              models={models}
              currentJob={currentJob}
              onRun={handleRunSegmentation}
              loading={loading}
              onNavigateToModels={() => setActiveTab('models')}
            />
            <MriViewer
              scan={scanData}
              volumes={volumes}
              hasOverlay={Boolean(currentJob && (currentJob.status === 'Completed' || currentJob.status === 'Demo mode'))}
              isDemo={isDemoActive}
            />
          </div>
        )}

        {activeTab === 'volumes' && scanData && (
          <VolumeTable
            volumes={volumes}
            isDemo={isDemoActive}
            voxelVolumeMm3={scanData.voxel_volume_mm3}
          />
        )}

        {activeTab === 'qc' && scanData && (
          <QualityControlPanel qcReport={scanData.qc_report} />
        )}

        {activeTab === 'evaluation' && scanData && (
          <EvaluationPanel
            scanId={scanData.scan_id}
            hasOverlay={Boolean(currentJob && (currentJob.status === 'Completed' || currentJob.status === 'Demo mode'))}
            evaluation={evaluation}
            onEvaluateSample={handleEvaluateSample}
            onEvaluateUpload={handleEvaluateUpload}
            loading={loading}
          />
        )}

        {activeTab === 'reports' && scanData && (
          <ReportsView
            scan={scanData}
            volumes={volumes}
            job={currentJob}
            evaluation={evaluation}
            isDemo={isDemoActive}
          />
        )}

        {activeTab === 'models' && (
          <ModelsView models={models} />
        )}

        {activeTab === 'about' && (
          <AboutLimitations />
        )}
      </main>

      <footer style={{
        borderTop: '1px solid var(--border-color)',
        padding: '20px 28px',
        textAlign: 'center',
        fontSize: '0.78rem',
        color: 'var(--text-dim)',
        background: 'rgba(15, 23, 42, 0.95)'
      }}>
        Cortex AI Research Prototype v2.0 • Reliability-Aware Brain MRI Analysis • Educational / Research Use Only • Local-Only Processing (No Cloud Uploads)
      </footer>
    </div>
  );
};

export default App;
